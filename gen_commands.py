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


def gen_commands(dataset, scene):
    """
    Generate train, nvs render, and relighting commands.

    # Example train command:
    python train_ord.py \
        --config ./configs/single_light/ord.txt \
        --datadir ./data/dataset/ord/antman/test \
        --expname ord_antman

    # Example render command:
    python train_ord.py \
        --config ./configs/single_light/ord.txt \
        --datadir ./data/dataset/ord/antman/test \
        --expname ord_antman \
        --render_only 1 \
        --render_test 1 \
        --ckpt log/ord_antman-20230531-013113/checkpoints/ord_antman_10000.th

    # Example relighting command:
    python scripts/relight_ord.py \
        --config configs/relighting_test/ord_relight.txt \
        --batch_size 800 \
        --datadir ./data/dataset/ord/antman/test \
        --hdrdir ./data/dataset/ord/antman/test \
        --geo_buffer_path ./relighting/ord_antman \
        --ckpt log/ord_antman-20230531-013113/checkpoints/ord_antman_10000.th
    """
    # Get the latest checkpoint path.
    ckpt_path = get_latest_checkpoint_path(dataset, scene)
    print(f"{dataset}_{scene}: {ckpt_path}")

    # Train.
    train_cmd = (f"python train_ord.py "
                 f"--config ./configs/single_light/{dataset}.txt "
                 f"--datadir ./data/dataset/{dataset}/{scene}/test "
                 f"--expname {dataset}_{scene}")

    # Render.
    render_cmd = (f"python train_ord.py "
                  f"--config ./configs/single_light/{dataset}.txt "
                  f"--datadir ./data/dataset/{dataset}/{scene}/test "
                  f"--expname {dataset}_{scene} "
                  f"--render_only 1 "
                  f"--render_test 1 "
                  f"--ckpt {ckpt_path}")

    # Relighting.
    relight_cmd = (f"python scripts/relight_ord.py "
                   f"--config configs/relighting_test/{dataset}_relight.txt "
                   f"--batch_size 800 "
                   f"--datadir ./data/dataset/{dataset}/{scene}/test "
                   f"--hdrdir ./data/dataset/{dataset}/{scene}/test "
                   f"--geo_buffer_path ./relighting/{dataset}_{scene} "
                   f"--ckpt {ckpt_path}")

    return train_cmd, render_cmd, relight_cmd


def main():
    all_cmds = []
    for dataset, scene in _datasets_scenes:
        train_cmd, render_cmd, relight_cmd = gen_commands(dataset, scene)
        print("######################")
        print(f"{dataset}_{scene}:")
        print(train_cmd)
        print(render_cmd)
        print(relight_cmd)

        # Write render and relight command to a file.
        all_cmds.append(render_cmd)
        all_cmds.append(relight_cmd)

    with open("commands.txt", "w") as f:
        f.write("\n".join(all_cmds))


if __name__ == "__main__":
    main()
