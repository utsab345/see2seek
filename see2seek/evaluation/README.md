# see2seek/evaluation/

Parallel evaluation for ImageNav and zero-shot ObjectNav, plus reusable SR/SPL
metrics. `__init__.py` exports `NavigationMetrics` and `Evaluator`.

## Evaluator

`Evaluator(cfg, checkpoint_path, device=None, num_envs=None, obs_encoder_type=None)`

- Deep-copies the config, then **restores `env/encoder/policy` from the
  checkpoint** — so you never need to repeat ablation flags at eval time, but
  a CLI `--obs_encoder` mismatch with the checkpoint raises `ValueError`.
- `evaluate(split="val", task="imagenav", num_episodes=None, log_file=None,
  zero_pointgoal=False)`:
  - loads episodes via `RoboTHOREnv._load_episodes` (unique IDs enforced),
    truncates to `num_episodes`, shards across VecEnv workers;
  - ImageNav: per-image CLIP goal embeddings; ObjectNav: CLIP **text**
    embeddings per RoboTHOR category (`_get_category_map` supports `en`/`ne`
    names) — PointGoal is zeroed in both ObjectNav and `zero_pointgoal`;
  - invalid-start episodes are excluded from SR/SPL and reported in a JSON
    coverage file; worker death → `RuntimeError` (eval never masks crashes);
  - returns SR/SPL plus difficulty buckets **easy ≤ 3 m, medium ≤ 6 m,
    hard > 6 m** (also `*_n`, `requested_*_n`, `invalid_*_n`).

## Per-episode log line

```
[ep {n}] scene=FloorPlan_Val2_4 id=... success=True steps=42 collisions=3 spl=0.612 path=7.2 shortest=4.4
```

Consumed by `scripts/eval_difficulty_trajectory.py` and
`scripts/plot_evaluation.py` — keep the format stable.

## NavigationMetrics

`NavigationMetrics()` — standalone SR/SPL accumulator (Anderson et al. 2018
definition: `SPL = (1/N) Σ Sᵢ·Lᵢ/max(pᵢ,Lᵢ)`), with
`update/reset/sr/spl/num_episodes/summary()`. Empty data → 0.0. The parallel
`Evaluator` computes metrics inline (it needs per-bucket stats); this helper is
kept as a clean reusable unit for serial or custom loops.