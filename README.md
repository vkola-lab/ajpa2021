# Deep learning driven quantification of interstitial fibrosis in kidney biopsies 

This work is published in the _American Journal of Pathology_ (https://doi.org/10.1016/j.ajpath.2021.05.005).

# Introduction
This repository contains a PyTorch implementation of a deep learning framework for classification of trichrome-stained whole-slide images (WSIs). Our framework is based on combining the features learned at the global level of the WSI along with the ones learned from local high-resolution image patches from the WSI:

<p align="center">
<img src="https://github.com/vkola-lab/ajpa2021/blob/master/figures/framework.png" width="70%" height="70%">
</p>

We use WSIs obtained from The Ohio State University Wexner Medical Center (OSUWMC) and Kidney Precision Medicine Project (KPMP) respectively to evaluate the proposed model. see below:
<p align="center">
<img src="https://github.com/vkola-lab/ajpa2021/blob/master/figures/dataset.png" width="65%" height="65%">
</p>

The Class Activation Map (CAM) is used to visualize which regions located by the model contributes to fibrosis level. CAMs for using local with global features were compared to CAMs for using global features only, as a purpose of showing that combining local and global features is able to produce better prediction results. see below:
<p align="center">
<img src="https://github.com/vkola-lab/ajpa2021/blob/master/figures/cam.png" width="70%" height="70%">
</p>

The performance of the model on OSUWMC are reported in the form of roc curve:
<p align="center">
<img src="https://github.com/vkola-lab/ajpa2021/blob/master/figures/result.png" width="70%" height="70%">
</p>

Please refer to our paper for more details.

# Usage
## Data
We trained, validated and tested the framework using the OSUWMC dataset and KPMP dataset. The OSUWMC dataset is not available for public use. KPMP is a public dataset and the WSIs can be downloaded from their website (https://www.kpmp.org).

Download the raw data and store them in the folder structure like below:
```
data/
    OSUWMC/
      1.svs
      2.svs
    OSUMASK_MASK/
      1.png
      2.png
```
## Preprocessing
Step1. Split training, validation, and testing dataset and store them in text files.
```
filename1,label1
filename2,label2
...
```
Step2. Preprocess mask/annotations provided by pathologists if possible. The mask indicates interested regions on WSIs created by the pathologists. If masks are provided, we will only consider those regions instead of the whole slide image. 
```
python src/preprocess.py --m
```
Step3. Preprocess images at global level. It downsampes the WSIs to a fixed dimension.
```
python src/preprocess.py --g
```
Step4. Preprocess images at local level. It tiles the WSIs to patches at high resolution and discards background patches. 
<p align="center">
<img src="https://github.com/vkola-lab/ajpa2021/blob/master/figures/local1.png" width="20%" height="20%"> <img src="https://github.com/vkola-lab/ajpa2021/blob/master/figures/local2.png" width="20%" height="20%"> <img src="https://github.com/vkola-lab/ajpa2021/blob/master/figures/local3.png" width="20%" height="20%"> <img src="https://github.com/vkola-lab/ajpa2021/blob/master/figures/local4.png" width="20%" height="20%">
</p>

```
python src/preprocess.py --p
```
Masks, global and local data will be saved in:
```
data/
    OSUWMC_all.txt
    OSUWMC_train.txt
    OSUWMC_val.txt
    OSUWMC_test.txt
    OSUWMC/
      1.svs
    OSUMASK_MASK/
      1.png
    masks/
    globals/
    locals/    
```
## Training & Evaluation
Sequentially run the following scripts and store the model checkpoints and logging files under "outputs/saved_models" and "outputs/runs".
```
1. bash train_g.sh
2. bash train_g2l.sh
```
To evaluate the model. run
```bash test.sh```

# Requirements
<li> WSI software: PixelView (deepPath, Inc.) </li>
Major dependencies are:
<li> python </li>
<li> pytorch </li>
<li> openslide-python </li>
