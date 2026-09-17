# see2seek/utils/

Shared tooling: centralized config + dataset path validation (used by every
entry point) and one-off data pipelines for building/extending the dataset.

## Config & validation

| File | Role |
|---|---|
| `config.py` | Dataclass tree (`EnvConfig`, `EncoderConfig`, `PolicyConfig`, `PPOConfig`, `LoggingConfig`, `DataConfig`, `Config`), `load_config(yaml)` deep-merge, `validate_dataset_paths(cfg)` |

`load_config` merges YAML onto defaults and **silently drops unknown keys** —
validate a resolved config before long runs. `validate_dataset_paths` is called
by `train.py`/`vec_env.py` and aborts early with a precise missing-file list
(it also requires `embeddings.pt` unless you run with a goal-embedding
registry already built).

## Dataset pipelines

| File | Role | Notes |
|---|---|---|
| `generate_data.py` | Raw RoboTHOR `.json.gz` → ImageNav dataset (renders goal image + `optimal_goal_pose`) | **No argparse; hardcoded absolute paths** — edit before running |
| `image2vec.py` | Pre-cache CLIP goal embeddings → `embeddings.pt` (resumable, batched 32) | Keys `"{split}/images/{filename}"` |
| `augment_goal_angles.py` | Render goal objects from `NUM_ANGLES` divergent views + re-encode → augmented episodes + `embeddings.pt` | Winners: camera faces object, offsets move *around* it (~4:1 success) |
| `filter_episodes.py` | Drop near-trivial episodes (`shortest_path_length ≤ 1.5 m` default, ~18% of data) | Backup + dry-run + `--dump_dropped_ids` |
| `check_episode_difficulty.py` | Report/plot `shortest_path_length` distribution | 3-panel figure (hist/sorted/CDF) |

## Visualization / misc

| File | Role | Notes |
|---|---|---|
| `viz_shortest_path.py` | Overlay oracle shortest-path waypoints + start/goal poses on a top-down frame | one PNG per episode |
| `birds_eye_views.py` | Tracking third-party camera test during random rollout | **Dev/smoke script: hardcoded paths** (`DEBUG_EPISODES_PATH`) |

## Gotchas

- `generate_data.py` and `birds_eye_views.py` contain hardcoded absolute,
  machine-specific paths and were dev tools — treat them as reference
  implementations before re-using in CI or on another box.
- `filter_episodes.py` treats a missing `shortest_path_length` as 0.0, i.e.
  it gets dropped — keep the backup flag on until you've inspected
  `--dump_dropped_ids`.