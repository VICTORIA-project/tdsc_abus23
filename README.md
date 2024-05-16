# **T**umor **D**etection, **S**egmentation and **C**lassification Challenge on **A**utomatic **B**reast **U**ltra**S**ound 2023 (TDSC-ABUS 2023)

This is the oficial implementation of the ViCOROB Team challenge submission for the TDSC-ABUS 2023 MICCAI challenge. The description of the challenge can be found on the [grand challenge website](https://tdsc-abus2023.grand-challenge.org/).<br>

A more detailed description of the methods and results can be found in the paper "SAM-PR: Enhancing 3D Automated Breast Ultrasound Imaging Segmentation with Probabilistic Refinement of SAM", presented in the [17th International Workshop on Breast Imaging (IWBI 2024)](https://www.iwbi2024.org/).

## How to run YOLOv8:
https://docs.ultralytics.com/quickstart/

<!-- Structure of the repository-->
## Structure of the repository

```
.
├── README.md
├── data
│   ├── challenge_2023
│   │   ├── TRAIN (original)
│   │   │   ├── DATA
│   │   │   ├── MASKS
│   │   └── only_lesion
│   │       ├── image_mha
│   │       ├── label_mha
├── checkpoints (for SAMed)
│   ├── epoch_159.pth (LoRA pretrained on Med)
│   ├── sam_vit_b_01ec64.pth (SAM pretrained)
├── SAMed (cloned from original repo)