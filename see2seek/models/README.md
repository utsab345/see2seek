# see2seek/models/

Frozen perception encoders. Both backbones are **always** in eval mode and have
`train()` overridden to no-op, so gradients never flow into them — only the
features they emit are learnable downstream (`agents/`).

See `encoders/` for the concrete modules.

## Encoder contracts

| Encoder | Output | Used for |
|---|---|---|
| DINOv2 ViT-B/14 | CLS `(B,768)` + patch grid `(B,256,768)` | observation RGB |
| CLIP ViT-B/32 (openai) | goal `(B,512)` | ImageNav image goals **and** ObjectNav text goals (shared latent space = zero-shot) |
| CLIP ViT-B/32 (obs mode) | `(B,512)` | CLIP-as-observation baseline (`--obs_encoder clip`) |