"""See2Seek: zero-shot embodied navigation in RoboTHOR.

Frozen DINOv2 + CLIP encoders feed a recurrent PPO policy with episodic
spatial memory. Trained on ImageNav (image goals) and evaluated zero-shot
on ObjectNav (CLIP text-encoded object goals).
"""

__version__ = "0.1.0"
