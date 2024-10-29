# Adapted from https://github.com/huggingface/pytorch-image-models/blob/v1.0.9/validate.py

import argparse
import csv
import json
import time
from collections import OrderedDict
from pathlib import Path

import ai_edge_torch  # XXX: when importing tensorflow, without importing ai_edge_torch, timm data loader does not start for some reason
import numpy as np
import tensorflow as tf
import timm
import torch
import torch.nn as nn
import torch.nn.parallel
from timm.data import create_dataset, create_loader
from timm.utils import AverageMeter, accuracy

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", metavar="DIR", default="/data/imagenet", help="path to dataset")
parser.add_argument("tflite_model", help="Path to the TFLite model file")
parser.add_argument("--split", metavar="NAME", default="validation", help="dataset split (default: validation)")
parser.add_argument("--model", "-m", metavar="NAME", help="model architecture")
parser.add_argument(
    "-j", "--workers", default=1, type=int, metavar="N", help="number of data loading workers (default: 1)"
)
parser.add_argument("--log-freq", default=10, type=int, metavar="N", help="batch logging frequency (default: 10)")
parser.add_argument("-b", "--batch-size", default=1, type=int, metavar="N", help="mini-batch size (default: 1)")
parser.add_argument(
    "--results-file", default="", type=str, metavar="FILENAME", help="Output csv file for validation results (summary)"
)
parser.add_argument(
    "--results-format", default="csv", type=str, help="Format for results file one of (csv, json) (default: csv)."
)


class TfLiteModel:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # prepare quantization parameters
        self.input_scale = None
        self.input_zero_point = None
        input_details = self.interpreter.get_input_details()
        if input_details[0]["quantization"] != (0.0, 0):
            self.input_scale, self.input_zero_point = input_details[0]["quantization"]
            self.input_dtype = input_details[0]["dtype"]

        self.output_scale = None
        self.output_zero_point = None
        output_details = self.interpreter.get_output_details()
        if output_details[0]["quantization"] != (0.0, 0):
            self.output_scale, self.output_zero_point = output_details[0]["quantization"]

    def __call__(self, input):
        # quantize input
        if self.input_scale is not None:
            input = (input / self.input_scale) + self.input_zero_point
            input = input.astype(self.input_dtype)

        self.interpreter.set_tensor(self.interpreter.get_input_details()[0]["index"], input)
        self.interpreter.invoke()

        # dequantize output
        output = self.interpreter.get_tensor(self.interpreter.get_output_details()[0]["index"])
        if self.output_scale is not None:
            output = (output - self.output_zero_point) * self.output_scale

        return output


def validate(args):
    # if `--model` is not specified, use the model that is embedded in the tflite model
    if args.model is None:
        args.model = Path(args.tflite_model).stem

    model = TfLiteModel(args.tflite_model)

    pretrained_cfg = timm.create_model(args.model, pretrained=False).pretrained_cfg
    input_size = pretrained_cfg.get("test_input_size", pretrained_cfg["input_size"])
    crop_pct = pretrained_cfg.get("test_crop_pct", pretrained_cfg["crop_pct"])

    criterion = nn.CrossEntropyLoss()

    dataset = create_dataset(
        root=args.data_dir,
        name="",  # ImageFolder or ImageTar are used if None
        split=args.split,
        input_img_mode="RGB" if input_size[0] == 3 else "L",
    )

    loader = create_loader(
        dataset,
        input_size=input_size,
        batch_size=args.batch_size,
        interpolation=pretrained_cfg["interpolation"],
        mean=pretrained_cfg["mean"],
        std=pretrained_cfg["std"],
        num_workers=args.workers,
        crop_pct=crop_pct,
        crop_mode=pretrained_cfg["crop_mode"],
        device=torch.device("cpu"),
    )

    batch_time = AverageMeter()
    forward_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # warmup, reduce variability of first batch time
    input = torch.randn((args.batch_size,) + tuple(input_size))
    model(input.numpy())

    end = time.time()
    for batch_idx, (input, target) in enumerate(loader):
        end_forward = time.time()

        output = model(input.numpy())
        forward_time.update(time.time() - end_forward, input.size(0))

        output = torch.from_numpy(output)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.detach(), target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.log_freq == 0:
            print(
                "Test: [{0:>4d}/{1}]  "
                "Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  "
                "Forward Time: {forward_time.val:.3f}s ({forward_time.avg:.3f}s, {rate_avg:>7.2f}/s)  "
                "Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  "
                "Acc@1: {top1.val:>7.3f} ({top1.avg:>7.3f})  "
                "Acc@5: {top5.val:>7.3f} ({top5.avg:>7.3f})".format(
                    batch_idx,
                    len(loader),
                    batch_time=batch_time,
                    forward_time=forward_time,
                    rate_avg=input.size(0) / batch_time.avg,
                    loss=losses,
                    top1=top1,
                    top5=top5,
                )
            )

    top1a, top5a = top1.avg, top5.avg
    results = OrderedDict(
        model=args.model,
        top1=round(top1a, 4),
        top1_err=round(100 - top1a, 4),
        top5=round(top5a, 4),
        top5_err=round(100 - top5a, 4),
        img_size=input_size[-1],
        crop_pct=crop_pct,
        interpolation=pretrained_cfg["interpolation"],
    )

    print(
        " * Acc@1 {:.3f} ({:.3f}) Acc@5 {:.3f} ({:.3f})".format(
            results["top1"], results["top1_err"], results["top5"], results["top5_err"]
        )
    )

    return results


def main():
    args = parser.parse_args()

    results = validate(args)
    if args.results_file:
        write_results(args.results_file, results, format=args.results_format)

    # output results in JSON to stdout w/ delimiter for runner script
    print(f"--result\n{json.dumps(results, indent=4)}")


def write_results(results_file, results, format="csv"):
    with open(results_file, mode="w") as cf:
        if format == "json":
            json.dump(results, cf, indent=4)
        else:
            if not isinstance(results, (list, tuple)):
                results = [results]
            if not results:
                return
            dw = csv.DictWriter(cf, fieldnames=results[0].keys())
            dw.writeheader()
            for r in results:
                dw.writerow(r)
            cf.flush()


if __name__ == "__main__":
    main()
