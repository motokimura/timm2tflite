from pathlib import Path

import ai_edge_torch
import timm
import torch
from ai_edge_torch.quantize.pt2e_quantizer import PT2EQuantizer, get_symmetric_quantization_config
from ai_edge_torch.quantize.quant_config import QuantConfig
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e


def quantize_model(
    model,
    calib_data_dir,
    convert_to_tflite=True,
    calib_batch_size=1,
    calib_num_workers=1,
    data_config=None,
):
    if data_config is None:
        data_config = model.pretrained_cfg

    # prepare calibration dataset
    input_size = data_config.get("test_input_size", data_config["input_size"])
    crop_pct = data_config.get("test_crop_pct", data_config["crop_pct"])
    dataset = timm.data.create_dataset(
        root=Path(calib_data_dir).parent.as_posix(),  # e.g., /data/imagenet
        name="",
        split=Path(calib_data_dir).name,  # e.g., calib
        input_img_mode="RGB" if input_size[0] == 3 else "L",
        is_training=False,
    )
    loader = timm.data.create_loader(
        dataset,
        input_size=input_size,
        batch_size=calib_batch_size,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=calib_num_workers,
        crop_pct=crop_pct,
        crop_mode=data_config["crop_mode"],
        device=torch.device("cpu"),
    )
    assert len(loader) > 0, f"Calibration data is empty. Please check {calib_data_dir=}"

    # prepare quantizer
    quantizer = PT2EQuantizer().set_global(get_symmetric_quantization_config(is_per_channel=False, is_dynamic=False))

    dummy_input = torch.randn(calib_batch_size, *input_size)
    quant_model = capture_pre_autograd_graph(model, (dummy_input,))
    quant_model = prepare_pt2e(quant_model, quantizer)
    # calibration
    for input, _ in loader:
        quant_model(input)
    quant_model = convert_pt2e(quant_model, fold_quantize=False)

    if convert_to_tflite:
        quant_model = ai_edge_torch.convert(
            quant_model, (dummy_input,), quant_config=QuantConfig(pt2e_quantizer=quantizer)
        )

    return quant_model
