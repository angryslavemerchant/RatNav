"""Pull-then-destroy failsafe for unattended overnight runs.

    python vast/watchdog.py --max-hours 7 --poll 180

Rented instances bill by the second, so an overnight run needs something that
kills them without depending on anyone being awake. The instances already
self-destroy when `run_training.sh` finishes, but that path is not enough on
its own for two reasons:

* it takes the log with it, and the log is where the result is;
* it never fires at all if training crashes, hangs, or the boot half-fails --
  which are exactly the cases where a machine sits idle and billing.

So this polls every tracked instance and, when one has finished OR has simply
been alive too long, **pulls its results first and only then destroys it**.
The age limit is the part that matters: it is a hard stop that does not care
why the instance is still up.

Scoped, deliberately, to instances this repo launched -- `.vast/instances.json`
-- because the account runs other projects' machines concurrently and a
watchdog that reasons about "all instances" would eventually eat one.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / ".vast" / "instances.json"
LOG = ROOT / "runs" / "watchdog.log"
LAUNCH = ROOT / "vast" / "launch.py"

# Markers written by run_training.sh / onstart.sh.
DONE = ("RUN_COMPLETE", "AWAITING_PULL", "GATE_FAILED", "SELF_DESTROY")


def say(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run(args: list[str], timeout: int = 300) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace",
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def python() -> str:
    return sys.executable


def tracked() -> list[dict]:
    if not STATE.exists():
        return []
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def live_ids() -> set[int]:
    code, out = run([python(), str(LAUNCH), "status"])
    if code != 0:
        return set()
    return {int(m) for m in re.findall(r"^\s*(\d{6,})\s+", out, re.M)}


def ssh_target(iid: int) -> tuple[str, str, str] | None:
    code, out = run([python(), str(LAUNCH), "ssh", "--id", str(iid)])
    match = re.search(r"ssh://([^@]+)@([^:]+):(\d+)", out)
    return match.groups() if match else None


def remote(iid: int, command: str, timeout: int = 120) -> str:
    target = ssh_target(iid)
    if target is None:
        return ""
    user, host, port = target
    code, out = run(
        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
         "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=15",
         "-p", port, f"{user}@{host}", command],
        timeout=timeout,
    )
    return out


def harvest(iid: int, label: str) -> None:
    """Pull results, then destroy. Order matters and is the whole point."""
    say(f"  {iid} ({label}): pulling results")
    code, out = run([python(), str(LAUNCH), "pull", "--id", str(iid)], timeout=600)
    if code != 0:
        say(f"  {iid}: pull FAILED ({code}) -- destroying anyway, "
            f"results were also uploaded to wandb")
    # Save the training log too; the on-instance log is the only place the
    # per-iteration history and the grid-cell verdict are printed.
    for name in ("m8_a.log", "m8_b.log", "m8_c.log", "m8_d.log", "train.log"):
        text = remote(iid, f"cat /workspace/{name} 2>/dev/null")
        if text.strip():
            destination = ROOT / "runs" / f"{label}_{name}"
            destination.write_text(text, encoding="utf-8")
            say(f"  {iid}: saved {destination.name} ({len(text)} bytes)")
    say(f"  {iid}: destroying")
    run([python(), str(LAUNCH), "destroy", "--id", str(iid)])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-hours", type=float, default=7.0, dest="max_hours",
                   help="hard stop, whatever the instance is doing")
    p.add_argument("--poll", type=int, default=180, help="seconds between checks")
    p.add_argument("--grace", type=float, default=0.0,
                   help="minutes to wait after completion before destroying")
    args = p.parse_args()

    started = time.time()
    say(f"watchdog up: max {args.max_hours}h per instance, "
        f"polling every {args.poll}s, scoped to {STATE}")

    while True:
        records = tracked()
        if not records:
            say("no tracked instances remain; watchdog exiting")
            return 0

        alive = live_ids()
        for record in list(records):
            iid, label = record["id"], record.get("purpose", "?")
            if iid not in alive:
                say(f"  {iid} ({label}): already gone, forgetting it")
                remaining = [r for r in tracked() if r["id"] != iid]
                STATE.write_text(json.dumps(remaining, indent=2), encoding="utf-8")
                continue

            created = record.get("created", "")
            age = float("nan")
            try:
                age = (time.time() - time.mktime(
                    time.strptime(created, "%Y-%m-%dT%H:%M:%SZ"))) / 3600.0
                age -= time.timezone / 3600.0  # created is UTC
            except (ValueError, TypeError):
                age = (time.time() - started) / 3600.0

            if age > args.max_hours:
                say(f"  {iid} ({label}): AGE LIMIT {age:.1f}h > "
                    f"{args.max_hours}h -- hard stop")
                harvest(iid, label)
                continue

            log = remote(iid, "tail -40 /workspace/*.log 2>/dev/null")
            if any(marker in log for marker in DONE):
                say(f"  {iid} ({label}): finished at {age:.1f}h")
                if args.grace:
                    time.sleep(args.grace * 60)
                harvest(iid, label)
            else:
                say(f"  {iid} ({label}): running, {age:.1f}h elapsed")

        time.sleep(args.poll)


if __name__ == "__main__":
    raise SystemExit(main())
