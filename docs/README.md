# docs/

Supporting artifacts for the README — no code.

| File | What it is |
|---|---|
| `system_architecture.png` | End-to-end pipeline diagram (DINOv2 patches/CLS, CLIP goal, episodic memory, GRU, actions) |
| `images/trajectory_*.png` | Sample bird's-eye trajectory rollouts per object/scene |

## Reading trajectory figures

- Green = successful trial, red = failed; light-green polyline = oracle
  shortest path.
- White circle w/ green ring = start; red circle w/ black ring = goal;
  green/red-ringed circles = agent's stop position (success/fail).

These visuals are produced by `scripts/visualize_trajectory.py`; regenerate
with newer checkpoints rather than shipping stale figures.