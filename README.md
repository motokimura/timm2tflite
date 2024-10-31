# timm2tflite

This repository is for converting and optionally quantizing models from [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) to TFLite using [ai-edge-torch](https://github.com/google-ai-edge).

As for quantization, models are firstly quantized with PyTorch [PT2E quantization](https://pytorch.org/tutorials/prototype/quantization_in_pytorch_2_0_export_tutorial.html) and then converted to TFLite with ai-edge-torch.
See [tools/quantization_utils.py](tools/quantization_utils.py) for the quantization process.

## Results

Randomly selected 5000 images from `val` set of ImageNet dataset are used for evaluation. Inference is done with CPU and batch size 1 for all models.

**Table 1: Top-1 accuracy (%) of the original PyTorch model and the converted TFLite model.**

model (from timm v1.0.9) | PyTorch fp32 | TFLite fp32
-- | -- | --
resnet18.a1_in1k | 73.94 | 73.94
convnextv2_tiny.fcmae_ft_in22k_in1k | 84.82 | 84.82
tf_efficientnetv2_s.in21k_ft_in1k | 84.42 | 84.42
efficientnet_lite0.ra_in1k | 75.44 | 75.44
mobilenetv4_conv_small.e2400_r224_in1k | 75.06 | 75.06
mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k.tflite | 83.04 | 83.04
maxvit_small_tf_224.in1k | 84.54 | 84.54
efficientformerv2_s0.snap_dist_in1k | 76.02 | 2.82

**Table 2: Top-1 accuracy (%) of PyTorch fp32, PyTorch int8, and TFLite int8 models.**

model (from timm v1.0.9) | PyTorch fp32 | PyTorch int8 | TFLite int8
-- | -- | -- | --
resnet18.a1_in1k | 73.94 | 68.70 | 67.96
efficientnet_lite0.ra_in1k | 75.44 | 68.20 | 62.90

## Environment Setup

```bash
docker compose build dev
docker compose run dev
```

## Prepare ImageNet Dataset (optional)

If you are going to quantize or evaluate the models, you need to prepare ImageNet dataset
(you can skip this section if you want to do simple TFLite conversion without quantization or evaluation).

<details close>
<summary>prepare ImageNet dataset</summary>

Download ImageNet dataset under `$HOME/data/imagenet`.

```bash
$ tree $HOME/data/imagenet -L 1

/home/motoki_kimura/data/imagenet
├── test
├── train
└── val
```

You may use [kaggle/imagenet-object-localization-challenge dataset](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data) to download ImageNet dataset.
After downloading the dataset, apply [valprep.sh](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh) to `val` directory to make it ImageFolder format.

```bash
mkdir -p $HOME/data/tmp_imagenet
cd $HOME/data/tmp_imagenet

# to run `kaggle competitions download`, you must first authenticate using an API token.
# see `Authentication` section in https://www.kaggle.com/docs/api
# or you can download the dataset manually from https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data
pip install kaggle
kaggle competitions download -c imagenet-object-localization-challenge

unzip imagenet-object-localization-challenge.zip

cd ILSVRC/Data/CLC-LOC/val
wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
sh valprep.sh
rm -f valprep.sh

mkdir -p $HOME/data/imagenet
cd $HOME/data/imagenet
mv $HOME/data/tmp_imagenet/ILSVRC/Data/CLC-LOC/train .
mv $HOME/data/tmp_imagenet/ILSVRC/Data/CLC-LOC/val .
mv $HOME/data/tmp_imagenet/ILSVRC/Data/CLC-LOC/test .

rm -rf $HOME/data/tmp_imagenet
```

You can remove `test` directory if you want to save disk space.

### Generate mini val dataset

To reduce evaluation time, you can use [tools/gen_mini_imagenet.py](tools/gen_mini_imagenet.py) to generate a subset of `val` dataset:

```bash
python tools/gen_mini_imagenet.py --data-dir /data/imagenet --out-dir /data/mini_imagenet_5000 --n-img-per-class 5
```

If you are evaluating with the mini dataset, please change `--data-dir /data/imagenet` to `--data-dir /data/mini_imagenet_5000` in the subsequent commands.

### Prepare calibration dataset for quantization

If you are going to quantize the models, use [tools/prep_calib.py](tools/prep_calib.py) to generate calibration dataset:

```bash
python tools/prep_calib.py /data/imagenet --n-img 512
```

With the command above, 512 images from `train` directory of ImageNet dataset will be copied to `$HOME/data/imagenet/calib` directory (`/data/imagenet/calib` in the container).

</details>

## Usage

### Convert to TFLite

```bash
python tools/tflite_export.py -m resnet18 --check-forward
```

Specify the model you want to convert with `-m` or `--model` option.
You can find the available model names from [pytorch-image-models/results/results-imagenet.csv](https://github.com/huggingface/pytorch-image-models/blob/main/results/results-imagenet.csv) etc.

You can check if the output of the PyTorch model and the TFLite model are consistent (whether the conversion was successful) by adding `--check-forward` option.

With the command above, `resnet18.a1_in1k.tflite` (TFLite-fp32 model) will be generated under the current directory.

### Evaluate TFLite models

(ImageNet dataset is required. See `Prepare ImageNet Dataset` section for details.)

After the conversion, you can evaluate the converted TFLite model on ImageNet:

```bash
python tools/tflite_validate.py resnet18.a1_in1k.tflite --data-dir /data/imagenet
```

### Convert to TFLite with quantization

(ImageNet dataset is required. See `Prepare ImageNet Dataset` section for details.)

You can generate a quantized TFLite model by adding `--quant` option:

```bash
python tools/tflite_export.py -m resnet18 --quant --calib-data-dir /data/imagenet/calib
```

`resnet18.a1_in1k.tflite` (TFLite-int8 model) will be generated under the current directory.

After the conversion, you can evaluate the quantized TFLite model on ImageNet:

```bash
python tools/tflite_validate.py resnet18.a1_in1k.tflite --data-dir /data/imagenet
```

### Evaluate PyTorch models (optionally with PT2E quantization)

(ImageNet dataset is required. See `Prepare ImageNet Dataset` section for details.)

To evaluate PyTorch models, use [tools/validate.py](tools/validate.py):

```bash
python tools/validate.py -m resnet18 --data-dir /data/imagenet
```

For evaluating PyTorch models, the default is to use GPU. You can also use CPU with `--device cpu` option.
Also, you can specify the batch size with `-b` or `--batch-size` option (note that the default batch size for evaluation of TFLite models is 1):

```bash
python tools/validate.py -m resnet18 --data-dir /data/imagenet --device cpu -b 1
```

To evaluate PyTorch models quantized with [PT2E](https://pytorch.org/tutorials/prototype/quantization_in_pytorch_2_0_export_tutorial.html), add `--quant` option:

```bash
python tools/validate.py -m resnet18 --data-dir /data/imagenet --device cpu -b 1 --quant --calib-data-dir /data/imagenet/calib
```
