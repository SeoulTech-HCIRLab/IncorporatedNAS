
- [Incorporated-NAS: An efficient Zero-shot proxy for light-weight NAS](#zen-nas-a-zero-shot-nas-for-high-performance-deep-image-recognition)
  - [Compare to Other Zero-Shot NAS Proxies on CIFAR-10/100](#compare-to-other-zero-shot-nas-proxies-on-cifar-10100)
  - [Reproduce Paper Experiments](#reproduce-paper-experiments)
    - [System Requirements](#system-requirements)
    - [Searching on CIFAR-10/100](#searching-on-cifar-10100)
    - [Searching on ImageNet](#searching-on-imagenet)
    - [Pretrain models](#Pretrain-models)
  - [Open Source](#open-source)


# Incorporated-NAS: An efficient Zero-shot proxy for light-weight NAS

This source code is the implementation of Incorporated-NAS. Our paper is published in the Electronics


## Compare to Other Zero-Shot NAS Proxies on CIFAR-10/100

We use the ResNet-like search space and search for models within the parameter budget 1M. All models are searched by the same evolutionary strategy, trained on CIFAR-10/100 for 1440 epochs with auto-augmentation, cosine learning rate decay, weight decay 5e-4. We report the top-1 accuracies in the following table:


| proxy              | CIFAR-10   | CIFAR-100  |
|--------------------|------------|------------|
| Incorporated-NAS-l | **96.66%**  | **80.67%** |
| Incorporated-NAS-s | **96.86%** | **81.1%**  |
| Zen-NAS            | 96.2%      | 80.1%      |
| FLOPs              | 93.1%      | 64.7%      |
| grad-norm          | 92.8%      | 65.4%      |
| synflow            | 95.1%      | 75.9%      |
| TE-NAS             | 96.1%      | 77.2%      |
| NASWOT             | 96.0%      | 77.5%      |
| Random             | 93.5%      | 71.1%      |

Please check our paper for more details.

## Reproduce Paper Experiments

### System Requirements

* PyTorch >= 1.5, Python >= 3.7
* By default, ImageNet dataset is stored under \~/data/imagenet; CIFAR-10/CIFAR-100 is stored under \~/data/pytorch\_cifar10 or \~/data/pytorch\_cifar100

### Searching on CIFAR-10/100
Searching for CIFAR-10/100 models with budget params < 1M , using different zero-shot proxies:

```bash

scripts/Combine_NAS_cifar_params1M.sh
scripts/Flops_NAS_cifar_params1M.sh
scripts/GradNorm_NAS_cifar_params1M.sh
scripts/NASWOT_NAS_cifar_params1M.sh
scripts/Params_NAS_cifar_params1M.sh
scripts/Random_NAS_cifar_params1M.sh
scripts/Syncflow_NAS_cifar_params1M.sh
scripts/TE_NAS_cifar_params1M.sh
scripts/Zen_NAS_cifar_params1M.sh

```

### Searching on ImageNet

Searching for ImageNet models:
```bash
scripts/CombineNAS_ImageNet_flops400M.sh
scripts/CombineNAS_ImageNet_flops600M.sh
```
### Pretrain models
All our pre-train EZenNet models [here!](https://drive.google.com/drive/folders/1DR5CLmI4pqhrwcyQ0pE2n8s3DjHiGbUs?usp=sharing)

## Open Source

A few files in this repository are modified from the following open-source implementations:

```text
https://github.com/idstcv/ZenNAS
https://github.com/DeepVoltaire/AutoAugment/blob/master/autoaugment.py
https://github.com/VITA-Group/TENAS
https://github.com/SamsungLabs/zero-cost-nas
https://github.com/BayesWatch/nas-without-training
https://github.com/rwightman/gen-efficientnet-pytorch
https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html
```
### Citing
If you find this work useful, please cite the following paper:
```text
@Article{electronics13163325,
AUTHOR = {Nguyen, Thi-Trang and Han, Ji-Hyeong},
TITLE = {Zero-Shot Proxy with Incorporated-Score for Lightweight Deep Neural Architecture Search},
JOURNAL = {Electronics},
VOLUME = {13},
YEAR = {2024},
NUMBER = {16},
ARTICLE-NUMBER = {3325},
URL = {https://www.mdpi.com/2079-9292/13/16/3325},
ISSN = {2079-9292},
DOI = {10.3390/electronics13163325}
}
```


