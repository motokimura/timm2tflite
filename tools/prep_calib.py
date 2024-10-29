import argparse
import random
import shutil
from pathlib import Path

parser = argparse.ArgumentParser(description="Prepare calibration data for static quantization")
parser.add_argument("data", metavar="DIR", help="path to dataset")
parser.add_argument(
    "--n-img", "-n", metavar="N", type=int, default=512, help="Number of the images used for caliblation"
)


def main():
    args = parser.parse_args()
    image_dir = Path(args.data) / "train"
    image_paths = list(image_dir.glob("*/*.JPEG"))

    random.seed(0)
    image_paths = random.sample(image_paths, args.n_img)

    out_dir = Path(args.data) / "calib"
    out_dir.mkdir()

    for src_path in image_paths:
        (out_dir / src_path.stem).mkdir(exist_ok=True)
        shutil.copy(src_path, out_dir / src_path.stem / src_path.name)


if __name__ == "__main__":
    main()
