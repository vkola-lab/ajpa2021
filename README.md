# Deep learning driven quantification of interstitial fibrosis in kidney biopsies 
# Introduction
This repository contains a PyTorch implementation of a deep learning framework for classification of trichrome-stained whole-slide images (WSIs). It is the code for the paper Deep learning driven quantification of interstitial fibrosis in kidney biopsies. Our framework is based on combining the features learned at the global level of the WSI along with the ones learned from local high-resolution image patches from the WSI:

<p align="center">
<img src="https://github.com/vkola-lab/ajpa2021/blob/master/figures/framework.png" width="70%" height="70%">
</p>

We use WSIs obtained from The Ohio State University Wexner Medical Center (OSUWMC) and Kidney Precision Medicine Project (KPMP) respectively to evaluate the proposed model. see below:
<p align="center">
<img src="https://github.com/vkola-lab/ajpa2021/blob/master/figures/dataset.png" width="70%" height="70%">
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
We trained, validated and tested the framework using the OSUWMC dataset and KPMP dataset. For OSUWMC dataset, please contact us for more information. KPMP is a public dataset and you could obtain WSIs from the official website (https://www.kpmp.org).

# Requirements
<li> OpenSlide </li>
