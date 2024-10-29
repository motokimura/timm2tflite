import argparse
from pathlib import Path

import ai_edge_torch
import numpy as np
import timm
import torch
from timm.utils.model import reparameterize_model

from tools.quantization_utils import quantize_model

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    "-m",
    metavar="MODEL",
    default="mobilenetv3_large_100",
    help="model architecture (default: mobilenetv3_large_100)",
)
parser.add_argument("--output", "-o", metavar="TFLITE_FILE", help="output model filename")
parser.add_argument(
    "--calib-data-dir", metavar="DIR", default="/data/imagenet/calib", help="path to the calibration dataset"
)
parser.add_argument(
    "--check-forward",
    action="store_true",
    default=False,
    help="Check forward pass of torch vs tflite models",
)
parser.add_argument("-b", "--batch-size", default=1, type=int, metavar="N", help="mini-batch size (default: 1)")
parser.add_argument("--reparam", "-r", default=False, action="store_true", help="Reparameterize model")
parser.add_argument("--quant", "-q", default=False, action="store_true", help="Quantize model")


def main():
    args = parser.parse_args()

    if args.quant:
        assert not args.check_forward, "`--check-forward` is not supported when `--quant` is specified"

    print(f"===> Creating PyTorch {args.model} model")
    model = timm.create_model(
        args.model,
        pretrained=True,
    )

    if args.output is None:
        # if `--output` is not specified, generate output filename from `pretrained_cfg`
        arch = model.pretrained_cfg["architecture"]  # e.g., "mobilenetv3_large_100"
        tag = model.pretrained_cfg["tag"]  # e.g., "ra_in1k"
        args.output = f"{arch}.{tag}.tflite"
    else:
        assert args.output.endswith(".tflite")
    args.output = Path(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.reparam:
        model = reparameterize_model(model)
    model.eval()

    input_size = model.pretrained_cfg.get("test_input_size", model.pretrained_cfg["input_size"])
    torch.manual_seed(0)
    dummy_input = torch.randn(args.batch_size, *input_size)
    torch_output = model(dummy_input)

    if args.quant:
        print("===> Quantizing and converting model to TFLite. It may take a while...")
        edge_model = quantize_model(model, args.calib_data_dir, convert_to_tflite=True)
    else:
        print("===> Converting model to TFLite. It may take a while...")
        edge_model = ai_edge_torch.convert(model.eval(), (dummy_input,))

    if args.check_forward:
        edge_output = edge_model(dummy_input)
        atol = 1e-5
        assert np.allclose(torch_output.detach().numpy(), edge_output, atol=atol)
        print(f"===> Forward pass check passed (atol={atol})")

    edge_model.export(args.output.as_posix())
    print(f"===> Successfully exported model to {args.output}")


if __name__ == "__main__":
    main()
