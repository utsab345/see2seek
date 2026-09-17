# configs/

Runtime YAML overrides for training and evaluation. These are **partial** configs:
each file is deep-merged on top of the dataclass defaults in
`see2seek/utils/config.py` (`Config`, `load_config`). Keys that do not exist on
the matching dataclass (top-level or one of `env`/`encoder`/`policy`/`ppo`/
`logging`/`data`) are silently ignored, so typos fail silently — always dump the
resolved config from a run to double-check.

## Files

| File | Stage | What it sets |
|---|---|---|
| `train_robothor.yaml` | training | `env.split=train`, dataset paths, checkpoint + log dirs |
| `eval.yaml` | evaluation | `env.split=val`, dataset paths, log dir |

## Usage

```bash
# Training (paths default to the repo-root dataset/train)
python scripts/train.py --config configs/train_robothor.yaml

# Evaluation
python scripts/eval.py --config configs/eval.yaml \
    --checkpoint data_dino_v7/checkpoints/checkpoint_final.pth
```

Every CLI entry point accepts `--config`; flags passed on the command line win
over YAML, which wins over defaults (see `scripts/` docs for per-arg names).

## Convention

Keep files minimal — only what differs from defaults. Dataset paths are the
most common override; if your data lives elsewhere, change
`env.scene_dataset_path` and `env.episodes_path` here rather than editing the
code, and re-run `validate_dataset_paths` (it runs automatically at startup and
aborts on missing files with a precise list).