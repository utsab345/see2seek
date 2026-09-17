# scripts/

Command-line entry points. All scripts insert the repository root on `sys.path`
and import `see2seek.*` lazily, so they can be run from anywhere with the repo
installed (`pip install -e .` or `PYTHONPATH=..`).

## Entry points

| Script | Purpose | Typical invocation |
|---|---|---|
| `train.py` | PPO training loop | `python scripts/train.py` |
| `eval.py` | Parallel ImageNav / zero-shot ObjectNav evaluation | `python scripts/eval.py --checkpoint data_dino_v7/checkpoints/checkpoint_final.pth` |
| `visualize_trajectory.py` | Bird's-eye trajectory rollouts over the AI2-THOR top-down map | `python scripts/visualize_trajectory.py --checkpoint ... --episodes FloorPlan_Val2_4_Bowl_2` |
| `eval_difficulty_trajectory.py` | Parse `eval_*.log` and report SR/SPL bucketed by geodesic difficulty | `python scripts/eval_difficulty_trajectory.py --log_file ...` |
| `plot_training.py` | 2×3 training-curve figure from `train_*.log` | `python scripts/plot_training.py --log_dir data_dino_v7/logs/train` |
| `plot_evaluation.py` | 2×3 evaluation figure from the latest `eval_*.log` | `python scripts/plot_evaluation.py --log_dir data_dino_v7/logs` |

The one-off data-wrangling utilities live under `see2seek/utils/`
(`generate_data.py`, `image2vec.py`, `augment_goal_angles.py`,
`filter_episodes.py`, `check_episode_difficulty.py`, `visualize` helpers).

## Gotchas

- `train.py` and `eval.py` accept both `--dashed` and `--underscored` forms for
  legacy flags; prefer the dashed form.
- Dataset paths are validated **before** W&B, model loading, and worker
  startup; feeds resolve `dataset/train/episodes` relative to the launch dir.
- `eval.py --task objectnav` zeroes the PointGoal input, so the (image-goal
  trained) policy transfers to text goals zero-shot via CLIP.
- Plots read the on-disk log format verbatim (regex-based). If you change the
  trainer's log line, update `plot_training.py` / `plot_evaluation.py` /
  `eval_difficulty_trajectory.py` regexes to match.