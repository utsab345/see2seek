"""Unit tests for the central configuration (see2seek.utils.config)."""

from pathlib import Path

import pytest
from see2seek.utils.config import Config, load_config, validate_dataset_paths


def test_default_policy_input_dim_is_2336() -> None:
    """Full dino pipeline: 1568 spatial + 64 CLS + 512 goal + 128 memory
    + 32 prev-action + 32 ego-pose."""
    assert Config().encoder.policy_input_dim == 2336


def test_ablation_input_dims() -> None:
    enc = Config().encoder

    enc.use_egopose = False
    assert enc.policy_input_dim == 2304

    enc.use_egopose = True
    enc.use_episodic_memory = False
    assert enc.policy_input_dim == 2208

    enc = Config().encoder
    enc.use_egopose = False
    enc.use_episodic_memory = False
    assert enc.policy_input_dim == 2176


def test_pointgoal_adds_32_dims() -> None:
    enc = Config().encoder
    enc.with_pointgoal = True
    assert enc.policy_input_dim == 2336 + 32


def test_clip_obs_branch() -> None:
    enc = Config().encoder
    enc.obs_encoder_type = "clip"
    enc.use_cls = False  # CLS irrelevant in clip-obs mode
    assert enc.policy_input_dim == (
        enc.clip_obs_proj_dim
        + enc.goal_proj_dim
        + enc.memory_proj_dim
        + enc.action_embed_dim
        + enc.egopose_embed_dim
    )


def test_load_config_deep_merges(tmp_path: Path) -> None:
    yaml_path = tmp_path / "overrides.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "env:",
                "  split: val",
                "logging:",
                "  use_wandb: false",
                "encoder:",
                "  use_egopose: false",
            ]
        ),
        encoding="utf-8",
    )
    cfg = load_config(str(yaml_path))
    assert cfg.env.split == "val"
    assert cfg.logging.use_wandb is False
    assert cfg.encoder.use_egopose is False
    assert cfg.ppo.lr == 2.5e-4  # untouched default


def test_load_config_ignores_unknown_keys(tmp_path: Path) -> None:
    yaml_path = tmp_path / "overrides.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "env:",
                "  split: train",
                "  not_a_real_field: 42",
                "invented_section:",
                "  x: 1",
            ]
        ),
        encoding="utf-8",
    )
    cfg = load_config(str(yaml_path))
    assert cfg.env.split == "train"


def test_validate_dataset_paths_raises_on_missing(tmp_path: Path) -> None:
    cfg = Config()
    cfg.env.scene_dataset_path = str(tmp_path / "does-not-exist")
    cfg.env.episodes_path = str(tmp_path / "episodes")
    with pytest.raises(FileNotFoundError):
        validate_dataset_paths(cfg, require_image_embeddings=False)


def test_validate_dataset_paths_ok_when_ready(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    episodes = scene / "episodes"
    episodes.mkdir(parents=True)
    (episodes / "ep_1.json.gz").write_text("{}", encoding="utf-8")
    (scene / "embeddings.pt").write_bytes(b"\x00")

    cfg = Config()
    cfg.env.scene_dataset_path = str(scene)
    cfg.env.episodes_path = str(episodes)
    validate_dataset_paths(cfg, require_image_embeddings=True)
    assert Path(cfg.env.scene_dataset_path).is_absolute()
