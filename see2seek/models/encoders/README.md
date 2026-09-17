# see2seek/models/encoders/

Frozen feature extractors. Never train these; they are the fixed cost of the
pipeline and dominate runtime.

## DINOv2Encoder (observation)

Frozen `dinov2_vitb14` loaded via `torch.hub` (a local clone at
`~/.cache/torch/hub/facebookresearch_dinov2_main` is picked up if present).
Constants: `embed_dim=768`, `num_patches=256`, `grid_size=16`.

- `get_all_embeddings(rgb) -> (cls (B,768), patches (B,256,768))` — the
  preferrered path; one backbone pass under fp16 autocast (CUDA), returns fp32.
- `forward(rgb) -> cls` and `get_patch_embeddings(rgb) -> patches` exist for
  backward compat. **Calling both back-to-back doubles ViT compute** — use
  `get_all_embeddings` to get CLS + patches in one forward.
- Inputs are `(B,3,H,W)`, ImageNet-normalized (`0.485…/0.229…`); H/W must be
  divisible by 14. Patch order is row-major rows of a 16×16 grid.
- `_shape_guard` raises `ValueError` on non-`(B,3,H,W)` input.

## CLIPGoalEncoder (goal)

Frozen CLIP ViT-B/32 (`openai`) via `open_clip`. `embed_dim=512`. All methods
are `@torch.no_grad()` and L2-normalize by default.

- `encode_image(...)` — PIL images (via CLIP `_preprocess`).
- `encode_text(...)` — strings, incl. Devanagari (e.g. `"कुर्सी"`); this is
  what powers zero-shot ObjectNav from RoboTHOR category names.
- `get_obs_embedding(rgb (B,3,224,224))` — CLIP obs mode; normalizes with CLIP
 's own mean/std (`0.4814…/0.2686…`) internally.
- Raises `ImportError` with an install hint if `open_clip_torch` is missing.

## Practical notes

- Goal embeddings are cached offline (`utils/image2vec.py`,
  `utils/augment_goal_angles.py`); at training time the CLIP encoder mostly
  runs only for text-goal transfers (`evaluation/evaluator.py`,
  `visualize_trajectory.py`).
- Both encoders are constructed with `device="cuda"` and moved explicitly;
  always call `encoder.get_all_embeddings(obs.to(device))` on the same device
  as the policy.