<p align="center">
  <img src="media/logo.jpeg" width="250"/>
</p>

# Solo Learn Radio - SLR
This repository is an adaption of the outstanding **solo-learn** library v1.0.7, bringing new features to produce benchmark on radio astronomical images and reproduce the paper "Self-supervised learning for radio-astronomy source classification: a benchmark" - Oral presentation at PRRS 2024 hosted by ICPR in Kolkata.

This work is being supported by ICSC.

## Contents
* General-purpose models for radio-astronomy
* Benchmark code
* Radio astronomical data classes
* Radio astronomical data processing techniques
* Experiment configurations

This repository provides links to models trained using various self-supervised learning (SSL) techniques on different pretraining datasets.
It provides che configurations used to reproduce experiments. Furthermore it provides the code for load and process radio-astronomical images.

If you use this repository please consider to cite the original work  **solo-learn [paper](#citation)**.

---

## News
* Added feature extraction
* Added runs summary
* Updated to include all4one method (as in the most recent version of solo-learn)
* Added k-fold cross validation (with fixed partition or generated at runtime)
* Added benchmark experiment generator
* Switch top5 to top2 accuracy and fix in case of 2 classes
* Added support for baselines
* Added online balancing strategy
* Added specific augmentations for radio images
* Added specific data classes for radio images

---

## Quickstart
* Clone the repo
* Obtain training/testing datasets (contact us: simone.riggi@inaf.it)
* Use a pretrained model
  * Evaluate with finetuning or linear evaluation
  * Feature extraction
  * Downstream task (e.g. classification)  

---

## Training methods tested

|      |      |
|--------------|--------------|
| [All4One](https://openaccess.thecvf.com/content/ICCV2023/html/Estepa_All4One_Symbiotic_Neighbour_Contrastive_Learning_via_Self-Attention_and_Redundancy_Reduction_ICCV_2023_paper.html)       | [BYOL](https://arxiv.org/abs/2006.07733)       |
| [SimCLR](https://arxiv.org/abs/2002.05709)      | [SwAV](https://arxiv.org/abs/2006.09882)      |
| [VICReg](https://arxiv.org/abs/2105.04906)       | [W-MSE](https://arxiv.org/abs/2007.06346)       |

---

## Backbones Trained
* [ResNet](https://arxiv.org/abs/1512.03385)

## Backbones Available
* [Swin](https://arxiv.org/abs/2103.14030)
* [PoolFormer](https://arxiv.org/abs/2111.11418)
* [WideResNet](https://arxiv.org/abs/1605.07146)
* [ViT](https://arxiv.org/abs/2010.11929)
* [ConvNeXt](https://arxiv.org/abs/2201.03545)

## Evaluation
* Offline k-fold cross validation
* Standard offline linear evaluation.
* Offline K-NN evaluation.
* Feature space visualization with UMAP.

## Logging
* Metric logging on the cloud with [WandB](https://wandb.ai/site)
* Custom model checkpointing with a simple file organization.

---

## Requirements
* torch
* torchvision
* tqdm
* einops
* wandb
* pytorch-lightning
* lightning-bolts
* torchmetrics
* scipy
* timm
* matplotlib
* seaborn
* pandas
* umap-learn

---

## Installation

First clone the repo.

Then, to install solo-learn with UMAP support, use:
```
pip3 install .[umap]
```

If no UMAP support is needed, the repository can be installed as:
```
pip3 install .
```

For local development:
```
pip3 install -e .[umap]
# Make sure you have pre-commit hooks installed
pre-commit install
```

**NOTE:** consider installing [Pillow-SIMD](https://github.com/uploadcare/pillow-simd) for better loading times

---

## Training

For pretraining the backbone, follow one of the many bash files in `scripts/pretrain/`.
We are now using [Hydra](https://github.com/facebookresearch/hydra) to handle the config files, so the common syntax is something like:
```bash
python3 main_pretrain.py \
    # path to training script folder
    --config-path scripts/pretrain/imagenet-100/ \
    # training config name
    --config-name barlow.yaml
    # add new arguments (e.g. those not defined in the yaml files)
    # by doing ++new_argument=VALUE
    # pytorch lightning's arguments can be added here as well.
```

After that, for offline linear evaluation, follow the examples in `scripts/linear` or `scripts/finetune` for finetuning the whole backbone.

For k-NN evaluation and UMAP visualization check the scripts in `scripts/{knn,umap}`.

---

## Evaluation and benchmark
```bash
sh _eval_pipeline.sh -d <dataset> -f <finetune<true|false>>
```
---

## Tutorials?

---

## Models available

Models pretrained on **curated dataset** and **resnet-18**:

| Method  | Checkpoint | MiraBest | RGZ | MSRS | VLASS |
|---------|:----------:|:--------:|:---:|:----:|:-----:|
| All4one | [60ffbwwo](https://drive.google.com/drive/folders/1B15fNDJkPBgS5UMPfbDQi_PFmPITRoiL?usp=sharing) | 82.1 ± 0.5 | **79.4 ± 0.2** | **78.2 ± 3.9** | **77.2 ± 0.8** |
| Byol    | [rfkiis97](https://drive.google.com/drive/folders/1NKHCs0BC68VliXzZBO-VAFjKWenPy4rp?usp=drive_link) | 89.6 ± 0.4 | 77.6 ± 0.2 | 78.0 ± 4.2 | 76.2 ± 0.6 |
| DINO    | [suikmnpg](https://drive.google.com/drive/folders/1jY5-rAVsYMj779VrFlN_iA64dzJq6ihO?usp=drive_link) | 64.2 ± 0.4 | 69.2 ± 0.6 | 73.8 ± 3.8 | 66.6 ± 0.8 |
| SimCLR  | [uiaz3umx](https://drive.google.com/drive/folders/1NUOKdxf2RYmbEcHVz_VFgKZdwfpY7Jqm?usp=drive_link) | **91.0 ± 0.5** | 69.5 ± 0.5 | 73.7 ± 4.1 | 71.9 ± 1.1 |
| SwAV    | [w6jni8a6](https://drive.google.com/drive/folders/1hBQiNdgg-2u5NDzBIPcmle2eL51Vm9Dm?usp=drive_link) | 72.9 ± 0.4 | 74.6 ± 0.7 | 74.8 ± 2.9 | 69.6 ± 1.3 |
| WMSE    | [pm6qc199](https://drive.google.com/drive/folders/1aGwnF5odDTptgHlxQBLf2eIQRMwaJfNl?usp=drive_link) | 84.6 ± 0.0 | 70.6 ± 0.5 | 74.6 ± 4.6 | 70.6 ± 0.0 |

Models pretrained on **curated dataset** and **resnet-50**:

| Method  | Checkpoint | MiraBest | RGZ | MSRS | VLASS |
|---------|:----------:|:--------:|:---:|:----:|:-----:|
| All4one | [hly1zogn](https://drive.google.com/drive/folders/1esVo28J_PPYqJkjIcBxKvtObq3iLQrlM?usp=drive_link) | 88.5 ± 0.6 | **78.8 ± 0.2** | **77.1 ± 3.6** | **77.0 ± 0.4** |
| Byol    | [xtulgzz6](https://drive.google.com/drive/folders/1R4mFNaFjAWA_O0EA9SYspCr2A8QDUf2y?usp=drive_link) | **90.0 ± 0.5** | 78.6 ± 0.5 | 76.5 ± 4.8 | 76.8 ± 0.4 |
| DINO    | [csn7idhy](https://drive.google.com/drive/folders/17hnFTc826gLJsEHdDM86coQMumlGBjR9?usp=drive_link) | 77.7 ± 1.1 | 70.2 ± 0.6 | 73.6 ± 3.5 | 67.7 ± 1.4 |
| SimCLR  | [4j8zhz46](https://drive.google.com/drive/folders/1iXkl8kAz8a4PDBxwowZXSlQmwz6u3DcZ?usp=drive_link) | 85.8 ± 0.9 | 73.0 ± 0.1 | 71.7 ± 4.2 | 73.1 ± 0.9 |
| SwAV    | [j1da6sjm](https://drive.google.com/drive/folders/1IdFiD0xOIFUbBdGpy3Veh0sPb2U2CAhv?usp=drive_link) | 82.3 ± 0.5 | 75.3 ± 0.2 | 74.4 ± 2.3 | 70.5 ± 0.5 |
| WMSE    | [pwfb6a5c](https://drive.google.com/drive/folders/1t9hrv5kbpDc_OWjH1n_jYc73athdJgsR?usp=drive_link) | 81.2 ± 0.5 | 74.7 ± 0.3 | 75.7 ± 4.4 | 72.6 ± 0.4 |

Models pretrained on **uncurated dataset** and **resnet-18**:

| Method  | Checkpoint | MiraBest | RGZ | MSRS | VLASS |
|---------|:----------:|:--------:|:---:|:----:|:-----:|
| All4one | [vdexi5ak](https://drive.google.com/drive/folders/1LghpB8_vynOnsqpnixN4RObnA2_FBGjW?usp=drive_link) | 75.6 ± 0.8 | 68.4 ± 0.5 | **74.2 ± 3.8** | **70.5 ± 0.4** |
| Byol    | [uja9qvb7](https://drive.google.com/drive/folders/12VRHUi8IXBeOTXKLQChDBz_cb02V3Djq?usp=drive_link) | 79.6 ± 0.4 | **72.6 ± 0.9** | 73.5 ± 3.0 | 69.6 ± 0.2 |
| DINO    | [g0x94tkz](https://drive.google.com/drive/folders/1Tv6vR98uNHxMy1j6gCoToHuB-H8bWq1x?usp=drive_link) | 83.6 ± 0.0 | 67.4 ± 1.0 | 71.4 ± 3.6 | 69.0 ± 0.8 |
| SimCLR  | [n11oftk9](https://drive.google.com/drive/folders/1QQD0eQbcANg6tDHyl-C_ZFL_tzzF5KNU?usp=drive_link) | **84.8 ± 0.7** | 68.2 ± 0.3 | 72.0 ± 3.4 | 68.1 ± 0.8 |
| SwAV    | [9xtqjn3m](https://drive.google.com/drive/folders/1vAGvlUv6W9OExNS1SSaPgtUIQyjp7JaX?usp=drive_link) | 74.4 ± 0.8 | 65.2 ± 0.9 | 73.2 ± 3.1 | 62.5 ± 0.6 |
| WMSE    | [w46gzvi8](https://drive.google.com/drive/folders/1vuu9T8jU-yIJfbHBUE7RZjTpXYtsRcas?usp=drive_link) | 64.6 ± 0.4 | 60.2 ± 0.5 | 69.6 ± 3.5 | 60.5 ± 0.6 |

### Models
Models are available using provided link on tables above, or you could download those via command line using sh scripts provided in utilities. The trained models should  be placed in trained_models folder, if not check your model path setting.

## Citation
If you use solo-learn, please cite [paper](https://jmlr.org/papers/v23/21-1155.html):
```bibtex
@article{JMLR:v23:21-1155,
  author  = {Victor Guilherme Turrisi da Costa and Enrico Fini and Moin Nabi and Nicu Sebe and Elisa Ricci},
  title   = {solo-learn: A Library of Self-supervised Methods for Visual Representation Learning},
  journal = {Journal of Machine Learning Research},
  year    = {2022},
  volume  = {23},
  number  = {56},
  pages   = {1-6},
  url     = {http://jmlr.org/papers/v23/21-1155.html}
}
```
