# see2seek/envs/

RoboTHOR simulation interface: a gym-style wrapper around the AI2-THOR
`Controller` with learned-reward shaping, plus a shared-memory multiprocessing
`VecEnv` that keeps RGB/observations off the worker IPC hot path.

## Modules

| Name | Role |
|---|---|
| `RoboTHOREnv` | Single-agent `Controller` wrapper: discrete actions, reward shaping, SPL bookkeeping, episode loading, crash/spawn retry |
| `VecEnv` | Spawn-context multiprocessing envs with persistent shared-memory buffers, auto worker respawn (train) / fail-fast (eval) |
| `make_vec_envs` | Default factory |

## Action space & observations

- `ACTIONS = {0: MoveAhead, 1: RotateLeft, 2: RotateRight, 3: Stop}`; grid
  `move_magnitude=0.25`, `rotate_degrees=30`, 224×224 RGB only.
- `reset()` returns `{"rgb", "goal", "pointgoal"}`; `step` returns
  obs/reward/done/info with SPL computed per episode
  (`L / max(p, L)`).

## Reward shape (for a fleet leader)

| Term | Value | When |
|---|---|---|
| success | +10.0 | stop within `success_distance` (1 m) |
| angle success | +5.0 | + within `angle_success_threshold` (25°) of goal heading |
| failed stop | −0.5 · min((d−1)/4, 1) | stop far from goal (if `shaped_stop`) |
| timeout | −2.0 | episode ends without Stop |
| geodesic delta | 1.5 · clip(Δgeo, −1, 1) | progress along shortest path |
| angle | 1.0 · Δangle | only inside success radius |
| collision | −0.01 | failed MoveAhead |
| rotation | −0.002 | every Rotate* |
| slack | −0.005 | per step |
| exploration | +0.10 | first visit of a 0.5 m cell (decays to 0.015) |

## Robustness (the parts future-you will thank you for)

- Controller hosting: GPU CloudRendering first with ~10 frame-probe passes,
  auto fallback to CPU renderer, per-worker port `8200+worker_id`, env
  `SEE2SEEK_NO_CLOUD_RENDERING` to force CPU.
- Invalid spawns are retried up to 20× per reset; controller crashes trigger a
  bounded restart (3×) that yields a synthetic terminal transition
  (`done=True`, `info["controller_crashed"]=True`) so training survives;
  eval **aborts** on worker death instead.
- Eval mode runs strict shards: `_reset_evaluation_env` skips
  `InvalidEpisodeStart`/exhausted shards and tracks `invalid_episodes`.
- Episodes load from `.json.gz` (preferred) or `.json`, from a dir or single
  file; `RoboTHOREnv._load_episodes(path, episode_ids=None)` is reused by the
  evaluator and visualizers.