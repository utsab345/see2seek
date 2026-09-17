# see2seek/agents/

Trainable control core: a 2-layer GRU Actor-Critic that fuses frozen encoder
features (DINOv2 CLS + spatial patches, CLIP goal) with previous action,
dead-reckoned ego-pose, optional PointGoal, and an episodic cross-attention
memory. This is the only part of the perception stack that learns (all
encoders are frozen).

## Modules

| Name | Role |
|---|---|
| `LinearNormAct` | `Linear → LayerNorm → ELU(inplace)` building block used in every branch head |
| `SpatialCompressionHead` | 2-layer CNN compressing the 16×16 DINOv2 patch grid → fixed 1568-dim vector, L2-normalized |
| `EpisodicMemory` | Single-head cross-attention over the last `memory_size` (128) CLS tokens **plus pose**; rolling buffer stores **detached** tokens (no BPTT through time) |
| `GRUActorCritic` | Fuses all branches → 2×GRU(512) → actor (4-actions) + critic (scalar) |
| `build_policy(cfg, device)` | Config-driven factory; raises `ValueError` on unknown encoder type / `memory_size < 1` |

## GRU input composition (default)

Spatial patches 1568 + CLS projection 64 + goal 512 + memory 128 +
prev-action 32 + ego-pose 32 = **2336**. PointGoal-on adds +32 → 2368.
Ablations remove branches: no-ego-pose → 2304, no-memory → 2208, neither → 2176
(the GRU re-allocates input size; these variants cannot resume each other's
checkpoints).

## Key behaviors

- **`forward`** runs a per-step loop with memory reset at episode boundaries
  (mask where `mask[t]=0`) and inserts the new CLS+pose into the rolling
  buffer each step.
- **`act(..., can_stop=None)`** masks the `Stop` logit to `-inf` until the
  environment allows stopping (`min_steps_before_stop`), preventing trivial
  early stopping.
- **`evaluate_actions`** recomputes log-probs/entropy from stored raw features
  for PPO; weight-init is orthogonal for GRU/conv/projection layers with small
  gains (`0.01`) on actor/critic heads.
- `SpatialCompressionHead` expects patch tokens in **row-major 16×16 grid
  order** — reshape is `view(B,16,16,768).permute(0,3,1,2)`, never a flat
  reshape.

## Construction

```python
from see2seek.utils.config import load_config
from see2seek.agents.gru_policy import build_policy

cfg = load_config("configs/train_robothor.yaml")
policy = build_policy(cfg, device="cuda")  # GRUActorCritic, ~?M params
```