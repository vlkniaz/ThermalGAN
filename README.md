<img src='imgs/dog.gif' align="right" width=384>

<br><br><br>

# ThermalGAN
This is the PyTorch implementation of the color-to-thermal image translation presented on ECCV 2018 in the paper .

The code is based on the PyTorch [implementation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) of the pix2pix and CycleGAN.

#### ThermalGAN: [[Project]](http://zefirus.org/ThermalGAN) [[Paper]](https://mula2018.github.io)
<img src="imgs/ThermalWorldVOC.jpg" width="900"/>

If you use this code for your research, please cite:

```
@InProceedings{Kniaz2018,
author={Kniaz, Vladimir V. and
Knyaz, Vladimir A. and
Hlad\r{u}vka, Ji{\v r}{\'{\i}}  and Kropatsch, Walter G. and Mizginov, Vladimir A.},
title={{ThermalGAN: Multimodal Color-to-Thermal Image Translation for Person Re-Identification in Multispectral Dataset}},
booktitle={{Computer Vision -- ECCV 2018 Workshops}},
year={2018},
publisher="Springer International Publishing",
}
```

## Prerequisites
- Linux or macOS
- Python 2 or 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation
- Install PyTorch and dependencies from http://pytorch.org
- Install Torch vision from the source.
```bash
git clone https://github.com/pytorch/vision
cd vision
python setup.py install
```
- Install python libraries [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate).
```bash
pip install visdom
pip install dominate
```
- Clone this repo:
```bash
git clone https://github.com/vlkniaz/ThermalGAN.git
```

### ThermalGAN train/test
- Download a ThermalGAN dataset:
```bash
bash ./datasets/download_thermalgan_dataset.sh thermalgan
```
- Train a model:
```bash
#!./scripts/train_thermalgan_rel.sh
python train.py --dataroot ./datasets/thermal_gan --name thermal_gan_rel --model thermal_gan_rel --which_model_netG unet_512 --which_direction AtoB --input_nc 4 --output_nc 1 --lambda_A 100 --dataset_mode thermal_rel --no_lsgan --norm batch --pool_size 0
```
- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097. To see more intermediate results, check out `./checkpoints/thermal_gan_rel/web/index.html`
- Test the model:
```bash
#!./scripts/test_thermalgan_rel.sh
python test.py --dataroot ./datasets/thermal_gan --name thermal_gan_rel --model thermal_gan_rel --which_model_netG unet_512 --which_direction AtoB --input_nc 4 --output_nc 1 --loadSize 512 --fineSize 512 --dataset_mode thermal_rel --how_many 352 --gpu_ids -1 --norm batch
```
The test results will be saved to a html file here: `./results/thermal_gan_rel/latest_test/index.html`.

### Apply a pre-trained model (ThermalGAN)

Download a pre-trained model with `./pretrained_models/download_thermalgan_dataset.sh`.

- For example, if you would like to download ThermalGAN model on the ThermalWorld dataset,
```bash
bash pretrained_models/download_thermalgan_model.sh ThermalGAN
```

- Download the ThermalWorld dataset
```bash
bash ./datasets/download_thermalworld_dataset.sh ThermalWorld
```
- Then generate the results using
```bash
bash scripts/test_thermalgan_rel_pretrained.sh
```
