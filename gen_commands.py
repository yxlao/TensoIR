from pathlib import Path

_datasets_scenes = [
    ("ord", "antman"),
    ("ord", "apple"),
    ("ord", "chest"),
    ("ord", "gamepad"),
    ("ord", "ping_pong_racket"),
    ("ord", "porcelain_mug"),
    ("ord", "tpiece"),
    ("ord", "wood_bowl"),
]


def get_latest_checkpoint_path(dataset, scene):
    """
    - In the latest timestamp folder
      - Find the largest iteration number, return that path

    Example checkpoint path:
    "log/ord_antman-20230530-181906/checkpoints/ord_antman_70000.th"
    """
    log_dir = Path("log")
    dataset_scene = f"{dataset}_{scene}"
    exp_dirs = [
        d for d in log_dir.iterdir() if d.is_dir() and dataset_scene in d.name
    ]
    if len(exp_dirs) == 0:
        raise ValueError(f"No experiment directory found for {dataset_scene}.")
    exp_dir = sorted(exp_dirs)[-1]
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_paths = [
        p for p in ckpt_dir.iterdir() if p.is_file() and p.suffix == ".th"
    ]
    if len(ckpt_paths) == 0:
        raise ValueError(f"No checkpoint found for {dataset_scene}.")
    ckpt_path = sorted(ckpt_paths)[-1]
    return ckpt_path


def gen_commands():
    pass


def main():
    """
    Generate train, nvs render, and relighting commands.
    """
    # Train
    # python train_ord.py \
    #     --config ./configs/single_light/ord.txt \
    #     --datadir ./data/dataset/ord/antman/test \
    #     --expname ord_antman

    for datset, scene in _datasets_scenes:
        # Get the latest checkpoint path
        ckpt_path = get_latest_checkpoint_path(datset, scene)
        print(f"{datset}_{scene}: {ckpt_path}")


if __name__ == "__main__":
    main()
