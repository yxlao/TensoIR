from pathlib import Path
import json
import shutil
import camtools as ct

script_dir = Path(__file__).parent.absolute()


def prepare_ord_relight():
    with open(script_dir / "ord_relight.json", "r") as f:
        ord_relight_list = json.load(f)

    # Check.
    for eval_item in ord_relight_list:
        gt_path = Path(eval_item["gt_path"])
        pd_src_path = Path(eval_item["pd_src_path"])
        if not gt_path.exists():
            print(f"{gt_path} does not exist.")
        if not pd_src_path.exists():
            print(f"{pd_src_path} does not exist.")

    # Prepare.
    # Copy pd_src_path -> pd_dst_path, mkdir if not exists.
    for eval_item in ord_relight_list:
        pd_src_path = Path(eval_item["pd_src_path"])
        pd_dst_path = Path(eval_item["pd_dst_path"])
        pd_dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(pd_src_path, pd_dst_path)
        print(f"Copy {pd_src_path} -> {pd_dst_path}")


def prepare_ord_nvs():
    with open(script_dir / "ord_nvs.json", "r") as f:
        ord_nvs_list = json.load(f)

    # Check.
    for eval_item in ord_nvs_list:
        gt_path = Path(eval_item["gt_path"])
        pd_src_path = Path(eval_item["pd_src_path"])
        if not gt_path.exists():
            print(f"{gt_path} does not exist.")
        if not pd_src_path.exists():
            print(f"{pd_src_path} does not exist.")

    # Prepare.
    # Copy pd_src_path -> pd_dst_path, mkdir if not exists.
    # Instead of simple copying, we need to crop the image by the left 1/3.
    for eval_item in ord_nvs_list:
        pd_src_path = Path(eval_item["pd_src_path"])
        pd_dst_path = Path(eval_item["pd_dst_path"])

        pd_dst_path.parent.mkdir(parents=True, exist_ok=True)
        im_pd = ct.io.imread(pd_src_path)
        src_shape = im_pd.shape
        im_pd = im_pd[:, :src_shape[1] // 3, :]
        dst_shape = im_pd.shape
        ct.io.imwrite(pd_dst_path, im_pd)
        print(f"Copy {pd_src_path} -> {pd_dst_path}, "
              f"shape: {src_shape} -> {dst_shape}")


def main():
    eval_nvs_dir = script_dir.parent / "eval_nvs"
    eval_relight_dir = script_dir.parent / "eval_relight"
    if eval_nvs_dir.is_dir():
        print(f"Removing {eval_nvs_dir}")
        shutil.rmtree(eval_nvs_dir)
    if eval_relight_dir.is_dir():
        print(f"Removing {eval_relight_dir}")
        shutil.rmtree(eval_relight_dir)

    prepare_ord_relight()
    prepare_ord_nvs()


if __name__ == "__main__":
    main()
