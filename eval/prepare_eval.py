from pathlib import Path
import json

script_dir = Path(__file__).parent.absolute()


def prepare_ord_relight():
    with open(script_dir / "ord_relight.json", "r") as f:
        ord_relight_list = json.load(f)
    for eval_item in ord_relight_list:
        gt_path = Path(eval_item["gt_path"])
        pd_src_path = Path(eval_item["pd_src_path"])
        if not gt_path.exists():
            print(f"{gt_path} does not exist.")
        if not pd_src_path.exists():
            print(f"{pd_src_path} does not exist.")


def prepare_ord_nvs():
    with open(script_dir / "ord_nvs.json", "r") as f:
        ord_nvs_list = json.load(f)
    for eval_item in ord_nvs_list:
        gt_path = Path(eval_item["gt_path"])
        pd_src_path = Path(eval_item["pd_src_path"])
        if not gt_path.exists():
            print(f"{gt_path} does not exist.")
        if not pd_src_path.exists():
            print(f"{pd_src_path} does not exist.")


def main():
    prepare_ord_relight()
    prepare_ord_nvs()


if __name__ == "__main__":
    main()
