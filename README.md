<div align="center">
<h1> 🩺 CMSA-Net </h1>
<h3>Causal Multi-scale Aggregation with Adaptive Multi-source Reference for Video Polyp Segmentation</h3>
<br>

<!-- [![Paper](https://img.shields.io/badge/Paper-Under_Review-gray)](#)  -->
[![Prediction Maps](https://img.shields.io/badge/Predictions-PVT--V2--B2-purple)](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/tong_wang_mbzuai_ac_ae/IgBMgKhbcJIfQqzZ5sC_TgKZAYXvAMg1GHbRVQeA21RtrKM?e=y6GGej)
[![Prediction Maps](https://img.shields.io/badge/Predictions-Res2Net50-orange)](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/tong_wang_mbzuai_ac_ae/IgCAy5pL4kWkTqqk9qUYMNR4Ac02RQX920el46GxIgxL33c?e=cJEbRd)

</div>

## 📢 News
- **[Feb, 2026]** CMSA-Net is currently under review. 
- **[Feb, 2026]** Released prediction results for both **PVT-V2-B2** and **Res2Net50** backbones.

<!-- ## 📌 Abstract
Video Polyp Segmentation (VPS) is a challenging task due to the varying size of polyps, motion blur, and the similarity between polyps and surrounding tissues. We propose **CMSA-Net**, a Contextual Multi-Scale Aggregation Network designed to capture robust spatio-temporal representations. By leveraging multi-scale feature fusion and contextual calibration, CMSA-Net achieves state-of-the-art performance on major benchmarks. -->

## 📊 Experimental Results

### Qualitative Results
<p align="center">
    <img src="https://github.com/wangtong627/CMSA-Net/blob/main/prediction/main_prediction.png" alt="Qualitative Results" width="90%">
</p>
<p align="center"><em>Figure: Qualitative comparison of CMSA-Net against other SOTA methods.</em></p>

---

### Quantitative Comparison on SUN-SEG Dataset
We evaluate our method on the **SUN-SEG** benchmark (Easy and Hard subsets). Our model (CMSA-Net) achieves the best performance across multiple metrics.

#### 1. SUN-SEG-Easy (Seen & Unseen)
| Method | Backbone | $S_\alpha\uparrow$ | $E_{\phi}^{mn}\uparrow$ | $F_\beta^w\uparrow$ | Dice $\uparrow$ | IoU$\uparrow$ | MAE$\downarrow$ | $S_\alpha\uparrow$ (U) | Dice$\uparrow$ (U) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| SEPNet (TCSVT'24) | PVTv2-B2 | 93.1 | 96.2 | 88.3 | 89.6 | 83.4 | 1.7 | 82.9 | 75.1 |
| STDDNet (ICCV'25) | PVTv2-B2 | 94.1 | 96.9 | 90.5 | 91.5 | 86.1 | 1.4 | 86.0 | 80.1 |
| **Ours (CMSA-Net)** | **Res2Net-50** | **94.5** | **97.3** | **90.5** | **91.9** | **86.5** | **1.2** | **84.4** | **77.5** |
| **Ours (CMSA-Net)** | **PVTv2-B2** | **95.1** | **97.5** | **91.6** | **92.6** | **87.6** | **1.1** | **86.7** | **80.3** |

#### 2. SUN-SEG-Hard (Seen & Unseen)
| Method | Backbone | $S_\alpha\uparrow$ | $E_{\phi}^{mn}\uparrow$ | $F_\beta^w\uparrow$ | Dice$\uparrow$ | IoU$\uparrow$ | MAE$\downarrow$ | $S_\alpha\uparrow$ (U) | Dice$\uparrow$ (U) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| SEPNet (TCSVT'24) | PVTv2-B2 | 89.4 | 94.0 | 83.5 | 85.7 | 77.6 | 3.4 | 84.7 | 77.4 |
| STDDNet (ICCV'25) | PVTv2-B2 | 91.1 | 95.0 | 86.0 | 87.8 | 80.6 | 2.8 | 86.3 | 80.2 |
| **Ours (CMSA-Net)** | **Res2Net-50** | **92.7** | **96.1** | **87.1** | **89.8** | **83.0** | **1.8** | **85.1** | **78.0** |
| **Ours (CMSA-Net)** | **PVTv2-B2** | **92.3** | **95.6** | **87.1** | **88.9** | **82.1** | **1.9** | **87.3** | **81.3** |

<small>*(U) indicates Unseen datasets. For full metrics, please refer to the paper.*</small>

## 📂 Download Prediction Results
As the paper is currently under review, the source code is not yet publicly available. However, we provide the full prediction maps for comparison and evaluation purposes:

* **PVT-V2-B2 Predictions:** [Download via OneDrive](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/tong_wang_mbzuai_ac_ae/IgBMgKhbcJIfQqzZ5sC_TgKZAYXvAMg1GHbRVQeA21RtrKM?e=y6GGej)
* **Res2Net50 Predictions:** [Download via OneDrive](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/tong_wang_mbzuai_ac_ae/IgCAy5pL4kWkTqqk9qUYMNR4Ac02RQX920el46GxIgxL33c?e=cJEbRd)
