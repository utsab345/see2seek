# see2seek/buffers/

Recurrent-PPO rollout storage. Two design points make this non-trivial:

1. It stores the **raw frozen-encoder outputs** (DINOv2 patches/CLS, CLIP goal)
   rather than the projected features, so `evaluate_actions` can re-run the
   trainable projections with gradients on every PPO epoch.
2. Rolling out a recurrent policy needs chunked mini-batches with the correct
   hidden state at each chunk boundary — plus, when episodic memory is on,
   memory snapshots at exactly those boundaries.

## Modules

| Name | Role |
|---|---|
| `RecurrentBatch` | Typed batch: chunked `patch_embeds/cls_embeds/goal_embeds/poses`, chunk-start `hidden_states`, `actions/old_log_probs/returns/advantages/masks`, PointGoal + dropout mask, optional memory snapshot triple |
| `RolloutBuffer` | Ring storage of `(T+1, N, …)` tensors, GAE returns, and `recurrent_mini_batches` chunking |

## Layout & storage

- Per-step tensors live on `storage_device` (default compute device) at
  `store_dtype` (default **fp16**); raw DINOv2 patches are the memory elephant
  (~0.8 GB per 2048-step update in fp16). When VRAM is tight set
  `ppo.buffer_storage_device: cpu`.
- `masks[t]=0` marks episode starts; hidden states are stored *before* each
  step so chunk-start hidden can be gathered exactly.
- `insert` detaches and casts inputs; `capture_memory` snapshots the episodic
  memory buffer only at chunk boundaries (kept in sync with chunk alignment).
- `after_update` rolls `hidden_states[0]`/`masks[0]` forward across updates —
  the recurrent state carries through updates via the stored trajectory, not
  PBT.

## Returns & batching

- `compute_returns(last_value, gamma=0.99, gae_lambda=0.95)` implements
  mask-aware GAE and normalizes advantages `(x-mean)/(std+1e-8)`.
- `recurrent_mini_batches(num_mini_batches)` shuffles **chunk** indices, splits
  evenly, and validates that each returned batch is aligned with captured
  memory boundaries; requires `1 ≤ num_mini_batches ≤ total_chunks`.
- `num_steps % chunk_len == 0` is enforced in `__init__`
  (`chunk_len=min(chunk_len, num_steps)`), so a `chunk_len=32` storage is
  always consumable by any valid `num_mini_batches`.