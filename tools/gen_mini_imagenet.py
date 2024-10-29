import argparse
import random
from pathlib import Path

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", "-d", required=True, help="ImageNet root directory which contains train and val directories"
    )
    parser.add_argument("--out-dir", "-o", required=True, help="Output directory")
    parser.add_argument("--split", "-s", type=str, default="val", choices=["val", "train"], help="train or val")
    parser.add_argument("--n-img-per-class", "-n", type=int, default=1, help="Number of images per class")
    return parser.parse_args()


def main():
    args = parse_args()

    args.data_dir = Path(args.data_dir)
    args.out_dir = Path(args.out_dir)

    data_dir = args.data_dir / args.split
    # list up all sysnet directories
    dirs = [d for d in Path(data_dir).iterdir() if d.is_dir() and d.name.startswith("n") and len(d.name) == 9]
    dirs.sort()
    assert len(dirs) == 1000, f"expected 1000 synsets, but got {len(dirs)} in {data_dir}"

    # prepare output directories
    out_dir = args.out_dir / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    for d in dirs:
        (out_dir / d.name).mkdir(parents=True, exist_ok=True)

    # randomly pick up N image per class (sysnet directory) and copy them to the output directories
    random.seed(42)
    for d in tqdm(dirs):
        files = list(d.iterdir())
        files.sort()
        for f in random.sample(files, args.n_img_per_class):
            out_file = out_dir / d.name / f.name
            f.link_to(out_file)

    print(f"successfully generated mini ImageNet dataset in {out_dir}")


if __name__ == "__main__":
    main()
