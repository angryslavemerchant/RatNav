# Vast.ai training automation — SmallCore

Rent a GPU, health-check it, train, push results to wandb, destroy the
instance — driven from `vast/launch.py` on the local machine.

Ported 2026-07-25 from the NeocoreEpisodic setup. `BLUEPRINT.md` is the
portable spec and `OFFER_JUDGEMENT.md` the machine-selection ledger; both carry
operational history worth reading before renting anything.

## What is different about this project

- **There is no dataset.** SmallCore generates its own walks in-process from a
  20 KB committed JSON. The download, Drive-bank and jpeg-decode gate tests
  measure a path this project never takes and are skipped
  (`--skip download,bank,cpu`). `thresholds_smallcore.json` keeps only the
  broken-hardware floors.
- **Rent for throughput, not speed.** The model is ~50k parameters and
  kernel-launch bound on the sequential position recurrence, not FLOP bound.
  Measured locally: a faster card buys very little on one run. The win is
  running *many configurations at once*. Default profile is `cheap`.
- **`destroy --all` is scoped to this repo.** The account runs instances for
  other projects concurrently; `--all` destroys only what
  `.vast/instances.json` records. `--all-remote` is the unscoped version and
  must be asked for by name.

## One-time setup

`vast/secrets.env` (gitignored — this repo is PUBLIC) with:

```
VAST_API_KEY=...
WANDB_API_KEY=...
HF_TOKEN=...
```

## Commands

```bash
python vast/launch.py search                       # candidate offers
python vast/launch.py scan --n 3                   # bench 3 machines, suggest thresholds
python vast/launch.py launch --smoke               # tiny pipeline test (keep-alive)
python vast/launch.py launch                       # default: m2_train.py, 8000 iters
python vast/launch.py launch --train-script scripts/m1_position.py \
                             --train-args "--iters 3000"
python vast/launch.py status                       # live instances (all projects)
python vast/launch.py logs [--id ID]
python vast/launch.py pull [--id ID]               # copy runs/ back before destroying
python vast/launch.py destroy [--id ID | --all]
```

## Lifecycle

1. `launch.py` picks an offer (median price, never the cheapest — the bottom of
   the range over-samples lemons) and creates the instance.
2. Onstart clones this repo and runs `vast/onstart.sh`: installs
   `requirements.txt`, runs the health gate (a sick machine **destroys
   itself**), then starts `vast/run_training.sh` in tmux.
3. `run_training.sh` runs the training script, uploads checkpoints and figures
   to the wandb run, and **destroys the instance** on success. On failure it
   stays alive for inspection. `--keep-alive` disables auto-destroy.

## Monitoring

- wandb project `smallcore` — losses and eval metrics live; checkpoints under
  Artifacts.
- `python vast/launch.py logs` — markers: `ONSTART_BEGIN`,
  `BENCHMARK_JSON {...}`, `GATE_PASSED`/`GATE_FAILED`, `TRAIN_LAUNCHED`,
  `TRAIN_EXIT`, `RUN_COMPLETE`, `SELF_DESTROY`.
- Empty logs past ~8 minutes means a silent zombie, not a slow boot — destroy
  and relaunch on a different `machine_id` (see OFFER_JUDGEMENT.md).

## Training scripts

Any script under `scripts/` works as `--train-script`. They log to wandb with
`--wandb` and write `runs/<name>/{best,latest}.pt` plus `metrics.json`, with
`runs/LATEST` naming the current run — which is what the upload step reads.
