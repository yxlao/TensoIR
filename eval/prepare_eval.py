from pathlib import Path
import json

script_dir = Path(__file__).parent.absolute()


def main():
    with open(script_dir / "ord_relight.json", "r") as f:
        ord_dict = json.load(f)

    # Check gt_path and pd_src_path exist.
    for ord_item in ord_dict:
        gt_path = Path(ord_item["gt_path"])
        pd_src_path = Path(ord_item["pd_src_path"])
        if not gt_path.exists():
            print(f"{gt_path} not found.")


if __name__ == "__main__":
    main()
