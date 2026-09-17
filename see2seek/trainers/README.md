# see2seek/trainers/

PPO training loop for the recurrent GRU policy. Single class,
`PPOTrainer(cfg, resume=None)`, launched from `scripts/train.py`.

## Lifecycle

1. **Validate** — dataset paths (`validate_dataset_paths`), and on `--resume`
   that the ablation flags (`use_egopose`, `use_episodic_memory`) match the
   saved config; a mismatch raises `ValueError` because GRU input size changes
   with the branches.
2. **Rollout** — 16 envs × 128 steps; frozen encoder inference on the main
   process; per-step `policy.forward` → `Categorical.sample` → `env.step`.
   `masks[t]=0` on episode start triggers memory reset.
3. **Update** — `RolloutBuffer.recurrent_mini_batches` → clipped surrogate,
   value MSE, entropy bonus. PointGoal-dropout weighting uses
   `entropy_coef_gps_on=0.02` / `entropy_coef_gps_off=0.08` when dropout is
   active. Grad clip 0.5, Adam 2.5e-4 with `LinearLR` decay → 0.1.
4. **Schedule** — curriculum `max_steps` 150→500 over 3 M env steps;
   exploration bonus 0.10 → 0.015 floor over 10 M steps; both linear.
5. **Log/checkpoint** — every `log_interval` (10) updates to W&B + console;
   every `checkpoint_interval` (50k) env steps save
   `checkpoint_{steps:012d}.pth`; `--resume` re-loads policy/optimizer/lr
   scheduler/total_steps/cfg.

## Checkpoint format

Dict: `policy_state_dict`, `optimiser_state_dict`, `lr_scheduler_state_dict`,
`total_steps`, `num_updates`, `cfg`. Saved inside the run's `checkpoints/`
dir; `checkpoint_final.pth` on completion.

## Reading the log line

```
[100000 steps | update 100] policy_loss=... value_loss=... entropy=...
reward=... SR=... SPL=... avg_steps=... fps=... max_steps=...
```

`plot_training.py` parses this exact format (and the legacy variant without
`avg_steps`).

## Notes

- Ego-pose is dead-reckoned and only advanced on **successful** MoveAhead
  (2D rotation matrices otherwise); this is critical for collision correctness.
- Rollout storage knobs `ppo.buffer_storage_device` / `ppo.buffer_store_dtype`
  are forwarded to `RolloutBuffer` — set `cpu`/`fp16` when memory-limited.
- `train()` closes the VecEnv and W&B in a `finally` block — always safe on
  early exit from Ctrl-C/resume.