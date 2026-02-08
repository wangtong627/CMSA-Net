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

---

### Quantitative Comparison on SUN-SEG Dataset
Complete quantitative results on the **SUN-SEG** benchmark. **Bold** indicates the best performance.

#### Table 1. Quantitative comparison on SUN-SEG-Easy dataset.
| Method | Publication | Backbone | Sα↑ | Eφ↑ | Fβ↑ | Dice↑ | IoU↑ | MAE↓ | Sα↑(U) | Eφ↑(U) | Fβ↑(U) | Dice↑(U) | IoU↑(U) | MAE↓(U) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| COSNet | TPAMI'19 | - | 84.5 | 83.6 | 72.7 | 73.0 | 64.8 | 3.4 | 65.4 | 60.0 | 43.1 | 42.3 | 34.2 | 7.3 |
| PCSA | AAAI'20 | - | 85.2 | 83.5 | 68.1 | 70.9 | 60.4 | 3.9 | 68.0 | 66.0 | 45.1 | 45.0 | 35.3 | 7.8 |
| 2/3D | MICCAI'20 | - | 89.5 | 90.9 | 81.9 | 82.9 | 75.6 | 2.1 | 78.6 | 77.7 | 65.2 | 65.6 | 57.0 | 4.4 |
| PraNet | MICCAI'20 | R2-50 | 91.8 | 94.2 | 87.7 | 88.3 | 82.5 | 2.0 | 78.1 | 78.8 | 66.3 | 66.5 | 58.2 | 5.2 |
| ACSNet | MICCAI'20 | R-34 | 92.0 | 94.2 | 87.4 | 88.2 | 82.8 | 1.7 | 77.2 | 76.6 | 63.0 | 63.8 | 56.4 | 4.6 |
| SANet | MICCAI'21 | R2-50 | 91.6 | 93.3 | 86.6 | 87.2 | 82.0 | 1.8 | 75.0 | 72.8 | 59.0 | 59.3 | 52.4 | 5.2 |
| SEPNet | TCSVT'24 | PVTv2-B2 | 93.1 | 96.2 | 88.3 | 89.6 | 83.4 | 1.7 | 82.9 | 88.3 | 73.5 | 75.1 | 66.6 | 4.2 |
| PNS | MICCAI'21 | R2-50 | 90.6 | 91.0 | 83.6 | 84.1 | 78.3 | 2.0 | 76.7 | 74.4 | 61.6 | 61.8 | 54.5 | 4.8 |
| PNS+ | MIR'22 | R2-50 | 91.7 | 92.5 | 84.8 | 85.5 | 78.7 | 2.1 | 80.6 | 79.8 | 67.6 | 67.8 | 59.1 | 4.4 |
| MAST | Arxiv'24 | PVTv2-B2 | 92.5 | 96.2 | 87.8 | 89.3 | 82.7 | 1.6 | 83.2 | 89.4 | 74.9 | 77.0 | 67.4 | 3.7 |
| SALI | MICCAI'24 | PVTv2-B2 | 90.2 | 93.2 | 84.9 | 85.8 | 78.9 | 2.4 | 73.1 | 75.2 | 58.7 | 59.2 | 50.2 | 6.3 |
| SALI | MICCAI'24 | PVTv2-B5 | 90.7 | 93.7 | 85.1 | 86.2 | 79.6 | 2.2 | 77.1 | 82.1 | 64.6 | 65.6 | 56.8 | 5.5 |
| STDDNet | ICCV'25 | R2-50 | 93.5 | 96.0 | 89.7 | 90.5 | 85.0 | 1.5 | 81.7 | 83.0 | 72.1 | 72.4 | 64.3 | 3.7 |
| STDDNet | ICCV'25 | PVTv2-B2 | 94.1 | 96.9 | 90.5 | 91.5 | 86.1 | 1.4 | 86.0 | **90.3** | 78.6 | 80.1 | 72.4 | 3.4 |
| **Ours** | **-** | **R2-50** | 94.5 | 97.3 | 90.5 | 91.9 | 86.5 | 1.2 | 84.4 | 90.1 | 75.3 | 77.5 | 69.2 | 3.5 |
| **Ours** | **-** | **PVTv2-B2** | **95.1** | **97.5** | **91.6** | **92.6** | **87.6** | **1.1** | **86.7** | **90.3** | **79.3** | **80.3** | **72.6** | **2.9** |

#### Table 2. Quantitative comparison on SUN-SEG-Hard dataset.
| Method | Publication | Backbone | Sα↑ | Eφ↑ | Fβ↑ | Dice↑ | IoU↑ | MAE↓ | Sα↑(U) | Eφ↑(U) | Fβ↑(U) | Dice↑(U) | IoU↑(U) | MAE↓(U) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| COSNet | TPAMI'19 | - | 78.5 | 77.2 | 62.6 | 63.3 | 54.1 | 4.6 | 67.0 | 62.7 | 44.3 | 43.8 | 35.3 | 7.0 |
| PCSA | AAAI'20 | - | 77.2 | 75.9 | 56.6 | 58.5 | 47.9 | 5.7 | 68.2 | 66.0 | 44.2 | 45.0 | 35.1 | 8.0 |
| 2/3D | MICCAI'20 | - | 84.9 | 86.9 | 75.3 | 76.4 | 67.1 | 3.5 | 78.6 | 77.5 | 63.4 | 64.4 | 55.8 | 4.4 |
| PraNet | MICCAI'20 | R2-50 | 88.4 | 91.9 | 83.1 | 83.9 | 76.6 | 3.1 | 78.7 | 80.2 | 66.7 | 67.5 | 58.7 | 5.3 |
| ACSNet | MICCAI'20 | R-34 | 87.2 | 91.0 | 80.6 | 82.0 | 74.8 | 3.6 | 76.2 | 77.6 | 61.0 | 62.4 | 54.7 | 5.3 |
| SANet | MICCAI'21 | R2-50 | 87.4 | 90.5 | 81.0 | 82.0 | 74.8 | 3.3 | 75.3 | 73.6 | 59.0 | 59.5 | 52.7 | 5.5 |
| SEPNet | TCSVT'24 | PVTv2-B2 | 89.4 | 94.0 | 83.5 | 85.7 | 77.6 | 3.4 | 84.7 | 89.5 | 74.5 | 77.4 | 68.4 | 3.9 |
| PNS | MICCAI'21 | R2-50 | 87.0 | 89.2 | 78.7 | 79.6 | 72.1 | 3.3 | 76.7 | 75.5 | 60.9 | 61.5 | 53.9 | 5.0 |
| PNS+ | MIR'22 | R2-50 | 88.7 | 90.2 | 80.6 | 81.3 | 72.8 | 3.0 | 79.8 | 79.3 | 65.4 | 66.1 | 57.1 | 5.0 |
| MAST | Arxiv'24 | PVTv2-B2 | 89.2 | 94.2 | 83.2 | 85.3 | 76.7 | 2.6 | 85.6 | 91.3 | 77.2 | 79.9 | 70.8 | 3.1 |
| SALI | MICCAI'24 | PVTv2-B2 | 86.8 | 90.9 | 79.9 | 81.0 | 72.6 | 3.4 | 72.8 | 75.9 | 56.9 | 57.9 | 48.7 | 6.8 |
| SALI | MICCAI'24 | PVTv2-B5 | 86.6 | 91.0 | 79.7 | 81.0 | 72.9 | 3.8 | 76.5 | 81.3 | 62.0 | 63.6 | 54.7 | 5.7 |
| STDDNet | ICCV'25 | R2-50 | 91.3 | 95.2 | 86.9 | 88.1 | 81.0 | 2.3 | 83.4 | 85.6 | 74.1 | 75.0 | 67.3 | 3.7 |
| STDDNet | ICCV'25 | PVTv2-B2 | 91.1 | 95.0 | 86.0 | 87.8 | 80.6 | 2.8 | 86.3 | 90.2 | 78.1 | 80.2 | 72.2 | 3.5 |
| **Ours** | **-** | **R2-50** | **92.7** | **96.1** | **87.1** | **89.8** | **83.0** | **1.8** | 85.1 | 89.8 | 75.0 | 78.0 | 69.5 | 3.6 |
| **Ours** | **-** | **PVTv2-B2** | 92.3 | 95.6 | **87.1** | 88.9 | 82.1 | 1.9 | **87.3** | **91.0** | **79.6** | **81.3** | **73.7** | **2.9** |

<small>*(U) indicates Unseen datasets. All scores are in percentage (%).*</small>

---

## 📂 Download Prediction Results
As the paper is currently under review, the source code is not yet publicly available. However, we provide the full prediction maps for comparison and evaluation purposes:

* **PVT-V2-B2 Predictions:** [Download via OneDrive](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/tong_wang_mbzuai_ac_ae/IgBMgKhbcJIfQqzZ5sC_TgKZAYXvAMg1GHbRVQeA21RtrKM?e=y6GGej)
* **Res2Net50 Predictions:** [Download via OneDrive](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/tong_wang_mbzuai_ac_ae/IgCAy5pL4kWkTqqk9qUYMNR4Ac02RQX920el46GxIgxL33c?e=cJEbRd)
