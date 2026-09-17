# See2Seek

Zero-shot embodied navigation in **RoboTHOR** using frozen **DINOv2 + CLIP**
encoders with a recurrent **PPO** policy and episodic spatial memory. Trained
on ImageNav (image goals); transfers zero-shot to ObjectNav (CLIP text-encoded
object goals) with no additional training.

[![CI](https://github.com/utsab345/see2seek/actions/workflows/ci.yml/badge.svg)](https://github.com/utsab345/see2seek/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)]()
[![MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Highlights

- **Frozen, pluggable encoders** — DINOv2 ViT-B/14 for scene observations,
  CLIP ViT-B/32 for image *and* text goals (the text path is what enables
  zero-shot ObjectNav).
- **Recurrent Actor-Critic** — 2-layer GRU (512 hidden); layer 1 fuses
  multimodal perception, layer 2 handles temporal reasoning.
- **Episodic spatial memory** — 128-slot rolling buffer of CLS tokens with
  pose-conditioned cross-attention; no BPTT through time.
- **Collision-aware ego-pose** — dead-reckoned `[x, y, cosθ, sinθ]`, updated
  only on successful moves; combined with memory it enables loop detection and
  room escape.
- **Recurrent-PPO training** — fp16/CPU-offloaded rollout storage, chunked
  mini-batches, max-steps curriculum, exploration-bonus decay, resumable
  checkpoints, W&B logging.
- **Ablations first-class** — direct ego-pose and episodic memory are optional;
  disabled branches are stripped from the architecture and GRU input, and every
  variant is resumable and reproducible.

## Results

Trained for 10M steps. Difficulty is the oracle shortest-path distance:
easy ≤ 3 m, medium 3–6 m, hard > 6 m.

| Task | Overall SR | Overall SPL | Easy SR | Hard SR |
|------|-----------:|------------:|--------:|--------:|
| ImageNav | 15.4% | 0.094 | 23.1% | 5.7% |
| ObjectNav (zero-shot) | 17.0% | 0.108 | 27.4% | 5.1% |

ObjectNav's higher overall SR is driven almost entirely by easy episodes
(27.4% vs 23.1%). On medium and hard episodes it trails ImageNav
(8.2% / 5.1% vs 8.7% / 5.7%), i.e. the CLIP text→vision alignment helps at
close range but does not produce sustained goal-directed long-horizon
navigation.

## Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Configuration](#configuration)
- [Reward function](#reward-function)
- [Metrics](#metrics)
- [Dataset](#dataset)
- [Development](#development)
- [Trajectory examples](#trajectory-examples)
- [References](#references)

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

The [RoboTHOR](https://arxiv.org/abs/2004.06799) simulator (`ai2thor`) is
required at train/eval time but is **not** needed for the unit test suite.

## Quickstart

```bash
# Train (DINOv2 obs encoder, 16 parallel RoboTHOR workers)
python scripts/train.py

# Debug mode (2 envs, 2 updates, no W&B)
python scripts/train.py --debug

# Resume from checkpoint
python scripts/train.py --resume data_dino_v7/checkpoints/checkpoint_000010000000.pth

# Evaluate ImageNav polymer
python scripts/eval.py --checkpoint data_dino_v7/checkpoints/checkpoint_final.pth --task imagenav

# Zero-shot ObjectNav (CLIP text goal, no GPS)
python scripts/eval.py --checkpoint data_dino_v7/checkpoints/checkpoint_final.pth --task objectnav

# Bird's-eye trajectory visualization
python scripts/visualize_trajectory.py \
    --checkpoint data_dino_v7/checkpoints/checkpoint_final.pth \
    --episodes FloorPlan_Val3_2_Apple_6 \
    --episodes_path /path/to/val/episodes \
    --scene_dataset_path /path/to/val

# Plot training curves / evaluation results
python scripts/plot_training.py
python scripts/plot_evaluation.py
```

### Ego-pose and episodic-memory ablations

```bash
python scripts/train.py --no-egopose              # drop direct ego-pose input
python scripts/train.py --no-episodic-memory      # drop memory + its GRU input
python scripts/train.py --no-egopose --no-episodic-memory
```

Each ablation is a fresh run with a distinct architecture (parameter shapes
change). See [Ablation reference](#ablation-reference-gru-input).

## Architecture

![System Architecture](docs/system_architecture.png)

| Branch | Source | Output dim | Trainable |
|--------|--------|-----------:|:---------:|
| Spatial | DINOv2 patch tokens (256×768) → 2-layer CNN | 1568 | ✅ |
| CLS | DINOv2 CLS token → Linear/LN/ELU | 64 | ✅ |
| Goal | CLIP ViT-B/32 embedding → Linear/LN/ELU | 512 | ❌ (frozen) |
| Episodic memory | Cross-attention over last 128 CLS tokens + poses | 128 | ✅ |
| Prev action | Learned embedding | 32 | ✅ |
| Ego-pose | Dead-reckoned [x, y, cosθ, sinθ] → Linear+ReLU | 32 | ✅ |
| **GRU input** | | **2336** | |

**Recurrent core:** 2-layer GRU, 512 hidden. Layer 1 fuses multimodal
perception; layer 2 handles temporal reasoning and planning. All frozen
encoders (DINOv2, CLIP) — only the spatial CNN, projections, memory module,
and GRU are trainable.

**Episodic memory** stores the last 128 CLS tokens in a rolling buffer with
pose-conditioned cross-attention readout. Stored tokens are detached (no BPTT
through time), resets at episode boundaries, and gives the agent a
"have I been here before?" signal without explicit map construction.

**Ego-pose** is dead-reckoned from discrete actions and updated only on
successful moves (collision-aware), so it is immune to wall failures.

### Ablation reference (GRU input)

| DINOv2 variant (PointGoal off) | GRU input dim |
|---|---:|
| Full model | 2336 |
| No direct ego-pose | 2304 |
| No episodic memory | 2208 |
| Neither branch | 2176 |

Disabled branches are removed from the model and GRU input entirely. The
no-memory variant has no attention parameters or memory buffers; the GRU stays
recurrent. `--no-egopose` removes only the direct 32-dim branch, leaving pose
conditioning intact inside memory. Evaluation and trajectory visualization
restore the architecture from the checkpoint automatically — no flags needed.

## Project structure

```
see2seek/
├── scripts/                  # CLI entry points
│   ├── train.py              #   PPO training (config/resume/debug/ablation flags)
│   ├── eval.py               #   ImageNav / ObjectNav evaluation
│   ├── evaluate (eval_difficulty_trajectory.py)
│   ├── visualize_trajectory.py
│   └── plot_training.py, plot_evaluation.py
├── see2seek/                 # installable package
│   ├── agents/               #   GRU Actor-Critic + episodic memory
│   ├── buffers/              #   recurrent-PPO rollout storage
│   ├── envs/                 #   RoboTHOR env + shared-memory VecEnv
│   ├── models/encoders/      #   frozen DINOv2 / CLIP
│   ├── trainers/             #   PPO loop (curriculum, resume, W&B)
│   ├── evaluation/           #   parallel evaluator + SR/SPL metrics
│   └── utils/                #   config, dataset tooling, viz helpers
├── configs/                  # YAML overrides (optional; CLI flags win)
├── docs/                     # architecture diagram, trajectory figures
├── tests/                    # unit tests (no simulator required)
├── .github/workflows/ci.yml  # lint + format + types + tests
├── pyproject.toml            # packaging + ruff/black/mypy/pytest config
└── Makefile                  # dev conveniences (make lint / check / test)
```

## Configuration

Configuration is a dataclass tree in `see2seek/utils/config.py`; YAML files in
`configs/` deep-merge over the defaults, so an override file only needs the
fields you want to change. Unknown keys are ignored.

Key training defaults:

- 16 parallel RoboTHOR workers (shared-memory VecEnv with auto worker respawn)
- 128 steps/env = 2048 samples per PPO update
- PPO: 4 epochs, 2 mini-batches, clip 0.2, entropy 0.05
- Adam lr 2.5e-4 with linear decay
- Curriculum: max_steps 150 → 500 over 3M steps
- Exploration bonus 0.10 per new cell, decaying to a 0.015 floor over 10M steps

Data paths are validated **before** W&B, model loading, or worker startup.
ImageNav requires cached `embeddings.pt` for image goals when episodic memory
is disabled.

## Reward function

| Param | Value | Purpose |
|------|---:|---|
| `success_reward` | +10.0 | Stop within 1 m of goal |
| `angle_success_reward` | +5.0 | Stop within 1 m AND facing goal heading (±25°) |
| `failed_stop_penalty` | −0.5 | Stop far from goal (shaped by distance) |
| `timeout_penalty` | −2.0 | Episode times out without stopping |
| `geodesic_reward_scale` | 1.5 | Reward for reducing shortest-path distance |
| `slack_reward` | −0.005 | Per-step cost |
| `exploration_bonus` | 0.10 | Intrinsic reward for new grid cells, decays to 0.015 over 10M steps |
| `collision_penalty` | −0.01 | Walking into walls |
| `rotation_penalty` | −0.002 | Per-rotation cost to prevent spinning |

Angle-to-goal shaping is active only within 1 m of the goal, encouraging
orientation before `Stop`.

## Metrics

- **SR (Success Rate)** — fraction of episodes where the agent calls `Stop`
  within 1 m of the goal.
- **SPL (Success weighted by Path Length)** (Anderson et al., 2018) — SR
  down-weighted by path efficiency: `SPL = S·L / max(p, L)`.

## Dataset

The dataset is **not** bundled. Build it with the tooling under
`see2seek/utils/`:

- `generate_data.py` — scrape/mint RoboTHOR episodes with shortest paths
- `filter_episodes.py` — drop degenerate episodes (e.g. shortest path ≤ 1.5 m,
  ~18% of the original set)
- `augment_goal_angles.py` — augment goal headings
- `image2vec.py` — pre-cache CLIP goal embeddings

Then point `env.scene_dataset_path` and `env.episodes_path` at it (or pass
`--scene-dataset-path` / `--episodes-path`) so `validate_dataset_paths`
succeeds.

## Development

```bash
make lint     # ruff + black --check
make check    # mypy
make test     # pytest
make format   # black + ruff --fix
```

All four gates also run in CI (`.github/workflows/ci.yml`) and as pre-commit
hooks. Tests are pure-Python unit tests (config merging, metrics, difficulty
parsing) and need no simulator or GPU.

Commits follow folder-scoped conventional titles, e.g.
`feat(agents): ...`, `fix(envs): ...`, `docs(trainers): ...`. See
`CONTRIBUTING.md` before submitting changes.

## Trajectory examples

Green paths are successful trials, red failed. Light green is the oracle
shortest path. White/green circle = start, red/black circle = goal, green
circle = successful stop, red circle = failed stop.

| | |
|:---:|:---:|
| ![Bowl](docs/images/trajectory_FloorPlan_Val2_4_Bowl_2.png) Bowl | ![Laptop](docs/images/trajectory_FloorPlan_Val2_2_Laptop_5.png) Laptop |
| ![SprayBottle](docs/images/trajectory_FloorPlan_Val1_5_SprayBottle_4.png) SprayBottle | ![Mug](docs/images/trajectory_FloorPlan_Val2_2_Mug_5.png) Mug |
| ![HousePlant](docs/images/trajectory_FloorPlan_Val1_1_HousePlant_3.png) HousePlant | ![BasketBall](docs/images/trajectory_FloorPlan_Val3_5_BasketBall_4.png) BasketBall |
| ![Laptop](docs/images/trajectory_FloorPlan_Val2_2_Laptop_1.png) Laptop | ![Bowl](docs/images/trajectory_FloorPlan_Val3_4_Bowl_7.png) Bowl |

## References

- [ZSON: Zero-Shot Object-Goal Navigation](https://arxiv.org/abs/2206.12403)
- [EmbCLIP: Simple but Effective CLIP Embeddings for Embodied AI](https://arxiv.org/abs/2111.09888)
- [DINOv2: Learning Robust Visual Features](https://arxiv.org/abs/2304.07193)
- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [RoboTHOR: An Open Simulation-to-Real Embodied AI Platform](https://arxiv.org/abs/2004.06799)

## License

MIT — see [LICENSE](LICENSE).