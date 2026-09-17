# see2seek/

Core Python package. Layout mirrors the data flow: encoders produce frozen
features → `agents` fuse them into a recurrent policy → `envs` step the
simulator → `buffers` store rollouts → `trainers` run PPO → `evaluation`
scores the policy. `utils` holds config + data tooling.

```
agents/      GRU Actor-Critic + episodic cross-attention memory
buffers/     Recurrent-PPO rollout storage + chunked minibatches
envs/        RoboTHOR wrapper (reward shaping) + shared-memory VecEnv
models/      Frozen DINOv2 (obs) and CLIP (goal) encoders
trainers/    PPO training loop, curriculum, checkpointing
evaluation/  Parallel evaluator + SR/SPL metrics
utils/       Central config, dataset generation, embedding caching
```

## Public surface

- `agents.gru_policy.build_policy(cfg, device="cuda")` — construct a policy
  from `Config`.
- `envs.vec_env.make_vec_envs(cfg, num_envs=None, **kwargs)` — shared-memory
  parallel environments.
- `trainers.ppo_trainer.PPOTrainer(cfg, resume=None)` — full training run.
- `evaluation.evaluator.Evaluator(cfg, checkpoint_path, ...)` — parallel eval.
- `utils.config.Config / load_config(yaml) / validate_dataset_paths(cfg)` —
  single source of truth for hyperparameters.

The only non-empty `__init__.py` exports `NavigationMetrics` and `Evaluator`
from `evaluation`; the rest are namespace marks.

## Data flow (one PPO update)

1. `RoboTHOREnv.reset()` → RGB + goal; `VecEnv` broadcasts via shared memory.
2. `PPOTrainer` runs DINOv2 `get_all_embeddings` (CLS + 16×16 patches) and CLIP
   goal embedding on the main process.
3. `GRUActorCritic.forward` fuses patches (CNN), CLS (projection), goal,
   prev-action, ego-pose, and episodic memory; emits a `Categorical` over the
   4 actions.
4. `RolloutBuffer` keeps the raw encoder outputs (not projections) so trainable
   heads recompute features with gradients in `evaluate_actions`.
5. GAE returns → chunked mini-batches → multi-epoch PPO update.