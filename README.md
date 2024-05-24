# MM-UNet: A Novel Cross-Attention Mechanism between Modules and Scales for Brain Tumor Segmentation (MM-UNet)
This repository contains the code for MM-UNet introduced in the following paper MM-UNet: A Novel Cross-Attention Mechanism between Modules and Scales for Brain Tumor Segmentation (EAAI 2024). Please read our paper at the following link: [paper_address](https://www.sciencedirect.com/science/article/abs/pii/S0952197624007498), which has been accepted by Engineering Applications of Artificial Intelligence.
Chih-Wei Lin* and Zhongsheng Chen  (* Corresponding Author)

# Citation
```bash
If you find  MM-UNet useful in your research, please consider citing:
@article{LIN2024108591,
title = {MM-UNet: A novel cross-attention mechanism between modules and scales for brain tumor segmentation},
journal = {Engineering Applications of Artificial Intelligence},
volume = {133},
pages = {108591},
year = {2024},
issn = {0952-1976},
doi = {https://doi.org/10.1016/j.engappai.2024.108591},
url = {https://www.sciencedirect.com/science/article/pii/S0952197624007498},
author = {Chih-Wei Lin and Zhongsheng Chen},
keywords = {Multi-scale fusion, Non-local attention, Cross-attention, Brain tumor segmentation, Multi-modules},
abstract = {Brain tumor segmentation is an essential issue in medical image segmentation. However, it is still challenging to consider the relationship between modules and efficiently fuse the features between adjacent scales. In this paper, we propose a novel cross-attention network for brain tumor segmentation, namely multi-scales and multi-modules cross-attention UNet (MM-UNet), which contains two mechanisms, module and scale cross-attentions. The module cross-attention (MCA) strategy connects and exchanges global information between adjacent modules. The scale cross-attention (SCA) strategy has two policies, the scale-related non-local relationship module (SCASNR) and the scale-related channel-based relationship module (SCASCR), that fuses the information between adjacent scales to mix the multi-scale information. Experiments on well-known tumor datasets, BraTS 2020, which has 369 cases, and has been classified into training, validation, and testing sets with 17,576, 4395, and 5735 images, to evaluate the performance by segmenting three regions, the whole tumor area (WT), core tumor area (CT) and enhancing tumor area (ET). Moreover, we consider three numerical metrics, dice, precision, and Hausdorff metrics, and various visualization results to objectively evaluate and intuitively display the experimental results. The proposed model surpasses state-of-the-art methods and achieves 0.8519, 0.8889, and 1.2647 with a base version network in dice, precision, sensitivity, and Hausdorff metrics, respectively. Moreover, we demonstrate the visualization with segmentation results and heatmaps in various scenarios to present the robustness of the proposed network in each region.}
}
```

## 1. Environment

- Please prepare an environment with Ubuntu 16.04, with Python 3.6.5, PyTorch 1.4.0, and CUDA 10.1
- others: einops 0.4.1, hausdorff 0.2.6, imageio 2.15.0, pandas 1.1.5 and Pillow 0.4.0

## 2. Dataset
Brats2020 and Brats2019 Download link:(https://www.med.upenn.edu/cbica/brats/)

## 3. Preprocess
- Preprocess
```bash
python preprocessing.py 
```
## 3. Train/Test

- Train

```bash
python train.py 
```

- Test 

```bash
python test.py 
```
