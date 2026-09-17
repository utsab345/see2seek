# Changelog

All notable changes to this project are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); project follows
folder-scoped conventional commits.

## [Unreleased]

- Project restructure: modern `pyproject.toml` packaging, dev tooling
  (ruff / black / mypy / pre-commit), CI pipeline, and a unit test suite.

## [0.1.0] - 2026-09-17

Initial import of the research codebase.

### Added

- Frozen DINOv2 ViT-B/14 observation encoder and CLIP ViT-B/32 goal encoder.
- 2-layer GRU Actor-Critic policy with pose-conditioned episodic
  cross-attention memory (no BPTT through memory).
- Recurrent-PPO rollout buffer with fp16/CPU offload storage and chunked
  mini-batches.
- RoboTHOR environment wrapper with shaped reward, crash recovery, and a
  shared-memory multiprocessing VecEnv.
- PPO trainer with curriculum (max-steps ramp), exploration-bonus decay,
  W&B logging, and resumable checkpoints.
- Parallel evaluator for ImageNav and zero-shot ObjectNav with SR/SPL and
  difficulty-bucket breakdowns.
- CLI entry points: train, eval, trajectory visualization, training/eval plot
  tools, difficulty bucketing.