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

## Model Zoo

All pretrained models avaiable can be downloaded directly via the tables below or programmatically by running one of the following scripts ...
`zoo/hulk.sh`, `zoo/banner.sh` and `zoo/imagenet.sh`.

---

## Models available

Models pretrained on **curated dataset** and **resnet-18**:

| Method  | Checkpoint | MiraBest| RGZ        |   MSRS    |      VLASS |
|---------|:----------:|:-------:|:----------:|:---------:|:----------:|
| All4one | [:link:]() |82.1&plusmn;0.5 |82.1&plusmn;0.5 |82.1&plusmn;0.5 |82.1&plusmn;0.5 |
| Byol    | [:link:]() |82.1&plusmn;0.5 |82.1&plusmn;0.5 |82.1&plusmn;0.5 |82.1&plusmn;0.5 |
| DINO    | [:link:]() |82.1&plusmn;0.5 |82.1&plusmn;0.5 |82.1&plusmn;0.5 |82.1&plusmn;0.5 |
| SimCLR  | [:link:]() |82.1&plusmn;0.5 |82.1&plusmn;0.5 |82.1&plusmn;0.5 |82.1&plusmn;0.5 |
| SwAV    | [:link:]() |82.1&plusmn;0.5 |82.1&plusmn;0.5 |82.1&plusmn;0.5 |82.1&plusmn;0.5 |
| WMSE    | [:link:]() |82.1&plusmn;0.5 |82.1&plusmn;0.5 |82.1&plusmn;0.5 |82.1&plusmn;0.5 |

Models pretrained on **curated dataset** and **resnet-50**:

| Method  | Checkpoint | MiraBest| RGZ        |   MSRS    |      VLASS |
|---------|:----------:|:-------:|:----------:|:---------:|:----------:|
| All4one | [:link:]() |82.1&plusmn;0.5 |82.1&plusmn;0.5 |82.1&plusmn;0.5 |82.1&plusmn;0.5 |
| Byol    | [:link:]() |82.1&plusmn;0.5 |82.1&plusmn;0.5 |82.1&plusmn;0.5 |82.1&plusmn;0.5 |
| DINO    | [:link:]() |82.1&plusmn;0.5 |82.1&plusmn;0.5 |82.1&plusmn;0.5 |82.1&plusmn;0.5 |
| SimCLR  | [:link:]() |82.1&plusmn;0.5 |82.1&plusmn;0.5 |82.1&plusmn;0.5 |82.1&plusmn;0.5 |
| SwAV    | [:link:]() |82.1&plusmn;0.5 |82.1&plusmn;0.5 |82.1&plusmn;0.5 |82.1&plusmn;0.5 |
| WMSE    | [:link:]() |82.1&plusmn;0.5 |82.1&plusmn;0.5 |82.1&plusmn;0.5 |82.1&plusmn;0.5 |

Models pretrained on **uncurated dataset** and **resnet-18**:

| Method  | Checkpoint | MiraBest| RGZ        |   MSRS    |      VLASS |
|---------|:----------:|:-------:|:----------:|:---------:|:----------:|
| All4one | [:link:]() |82.1&plusmn;0.5 |82.1&plusmn;0.5 |82.1&plusmn;0.5 |82.1&plusmn;0.5 |
| Byol    | [:link:]() |82.1&plusmn;0.5 |82.1&plusmn;0.5 |82.1&plusmn;0.5 |82.1&plusmn;0.5 |
| DINO    | [:link:]() |82.1&plusmn;0.5 |82.1&plusmn;0.5 |82.1&plusmn;0.5 |82.1&plusmn;0.5 |
| SimCLR  | [:link:]() |82.1&plusmn;0.5 |82.1&plusmn;0.5 |82.1&plusmn;0.5 |82.1&plusmn;0.5 |
| SwAV    | [:link:]() |82.1&plusmn;0.5 |82.1&plusmn;0.5 |82.1&plusmn;0.5 |82.1&plusmn;0.5 |
| WMSE    | [:link:]() |82.1&plusmn;0.5 |82.1&plusmn;0.5 |82.1&plusmn;0.5 |82.1&plusmn;0.5 |

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
