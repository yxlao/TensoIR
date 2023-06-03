from pathlib import Path
import json
import shutil
import camtools as ct

script_dir = Path(__file__).parent.absolute()


def prepare_relight(json_path):
    with open(json_path, "r") as f:
        eval_items = json.load(f)

    # Check.
    found_invalid = False
    for eval_item in eval_items:
        gt_path = Path(eval_item["gt_path"])
        pd_src_path = Path(eval_item["pd_src_path"])
        if not gt_path.exists():
            found_invalid = True
            print(f"{gt_path} does not exist.")
        if not pd_src_path.exists():
            found_invalid = True
            print(f"{pd_src_path} does not exist.")
    if found_invalid:
        print("Aborted.")
        return

    # Prepare.
    # Copy pd_src_path -> pd_dst_path, mkdir if not exists.
    for eval_item in eval_items:
        pd_src_path = Path(eval_item["pd_src_path"])
        pd_dst_path = Path(eval_item["pd_dst_path"])
        pd_dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(pd_src_path, pd_dst_path)
        print(f"Copy {pd_src_path} -> {pd_dst_path}")


def prepare_nvs(json_path):
    with open(json_path, "r") as f:
        eval_items = json.load(f)

    # Check.
    found_invalid = False
    for eval_item in eval_items:
        gt_path = Path(eval_item["gt_path"])
        pd_src_path = Path(eval_item["pd_src_path"])
        if not gt_path.exists():
            found_invalid = True
            print(f"{gt_path} does not exist.")
        if not pd_src_path.exists():
            found_invalid = True
            print(f"{pd_src_path} does not exist.")
    if found_invalid:
        print("Aborted.")
        return

    # Prepare.
    # Copy pd_src_path -> pd_dst_path, mkdir if not exists.
    # Instead of simple copying, we need to crop the image by the left 1/3.
    for eval_item in eval_items:
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

    # prepare_relight(script_dir / "ord_relight.json")
    # prepare_nvs(script_dir / "ord_nvs.json")

    prepare_relight(script_dir / "synth4relight_relight.json")
    # prepare_nvs(script_dir / "synth4relight_nvs.json")


if __name__ == "__main__":
    main()
