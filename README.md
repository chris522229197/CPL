# Reproducing CPL: Counterfactual Prompt Learning for Vision and Language Models

This forked repository contains code for reproducing the EMNLP 2022 paper [CPL: Counterfactual Prompt Learning for Vision and Language Models](https://arxiv.org/abs/2210.10362). Here is the abstract of the paper:

> Prompt tuning is a new few-shot transfer learning technique that only tunes the learnable prompt for pre-trained vision and language models such as CLIP. However, existing prompt tuning methods tend to learn spurious or entangled representations, which leads to poor generalization to unseen concepts.
Towards non-spurious and efficient prompt learning from limited examples, this paper presents a novel Counterfactual Prompt Learning (CPL) method for vision and language models, which simultaneously employs counterfactual generation and contrastive learning in a joint optimization framework.
Particularly, CPL constructs counterfactual by identifying minimal non-spurious feature change between semantically-similar positive and negative samples that causes concept change and learns more generalizable prompt representation from both factual and counterfactual examples via contrastive learning.
Extensive experiments demonstrate that CPL can obtain superior few-shot performance on different vision and language tasks than previous prompt tuning methods on CLIP. On image classification, we achieve a 3.55% average relative improvement on unseen classes across seven datasets; on image-text retrieval and visual question answering, we gain up to 4.09% and 25.08% relative improvements across three few-shot scenarios on unseen test sets.

## Installation
1. This code is built on top of the PyTorch toolbox [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch), so you need to install the `dassl` environment first. To install the environment, follow the instructions described in the original codebase [here](https://github.com/KaiyangZhou/Dassl.pytorch#installation) to install the modified Dassal package in `CPL/Dassl.pytorch`.
2. With the `dassl` environment activated, install the requirements in `CPL/req.txt` using pip.


## Dataset preparation
- Follow [DATASETS.md](CPL/DATASETS.md) and download the datasets used.
- Follow instructions in the [dataset_processing](CPL/dataset_processing) directory to preprocess the image-text retrieval and visual question answering datasets.


## Reproducing the results

Before running any code you must activate the conda environment:

    source activate dassl


### Running arguments for training: 
You can launch the corresponding training experiments by passing the appropriate arguments to
```sh
cd CPL
python train.py 
```
for CoCoOp, or 
```sh
cd CPL
python train_cf.py 
```
for CPL.

The relevant arguments for training are:

| Parameter          | Description  |
| :----------------: | :------------|
| --root ${DATA_DIR} | Parent directory containing all dataset directories. |
| --seed ${SEED} | Integer random seed for training. |
| --trainer ${TRAINER} |'CoCoOp' for CoCoOp, and 'CoCoOpcf' for CPL. |
| --dataset-config-file configs/datasets/${DATASET}.yaml | Path for dataset configuration file. |
| --config-file configs/trainers/${TRAINER}/${CFG}.yaml | Path for model trainer configuration file. |
| --output-dir ${OUT_DIR} | Trained model output directory. |
| DATASET.NUM_SHOTS ${SHOTS} | Number of shots for image classification, ITR, and VQA. |
| DATASET.SUBSAMPLE_CLASSES ${MODE} | 'base' for image classification, and fewshot' for IRT and VQA. |

An example for training CoCoOp on OxfordPets:
```sh
cd CPL
python train.py \
    --root /path/to/datasets \
    --seed 1 \
    --trainer CoCoOp \
    --dataset-config-file configs/datasets/oxford_pets.yaml \
    --config-file configs/trainers/CoCoOp/vit_b16_c4_ep10_batch1_ctxv1.yaml \
    --output-dir /path/to/cocoop/model/output \
    DATASET.NUM_SHOTS 16 \
    DATASET.SUBSAMPLE_CLASSES base
```
An example for training CPL on OxfordPets:
```sh
cd CPL
python train_cf.py \
    --root /path/to/datasets \
    --seed 1 \
    --trainer CoCoOpcf \
    --dataset-config-file configs/datasets/oxford_pets.yaml \
    --config-file configs/trainers/CoCoOp/vit_b16_c4_ep10_batch1_ctxv1.yaml \
    --output-dir /path/to/cpl/model/output \
    DATASET.NUM_SHOTS 16 \
    DATASET.SUBSAMPLE_CLASSES base
```

### Running arguments for test evaluation: 
You can launch the corresponding test evaluation experiments by passing the appropriate arguments to
```sh
cd CPL
python train.py 
```
for CoCoOp, or 
```sh
cd CPL
python train_cf.py 
```
for CPL.

The relevant arguments for test evaluation are:

| Parameter          | Description  |
| :----------------: | :------------|
| --root ${DATA_DIR} | Parent directory containing all dataset directories. |
| --seed ${SEED} | Integer random seed used during training. |
| --trainer ${TRAINER} |'CoCoOp' for CoCoOp, and 'CoCoOpcf' for CPL. |
| --dataset-config-file configs/datasets/${DATASET}.yaml | Path for dataset configuration file. |
| --config-file configs/trainers/${TRAINER}/${CFG}.yaml | Path for model trainer configuration file used during training. |
| --model-dir ${MODEL_DIR} | Directory containing the training outputs. |
| --output-dir ${OUT_DIR} | Test evaluation result output directory. |
| --load-epoch ${EPOCH} | Number of epochs used to trained the model. |
| --eval-only | Flag for running the script in evaluation mode. |
| DATASET.NUM_SHOTS ${SHOTS} | Number of shots for image classification, ITR, and VQA. |
| DATASET.SUBSAMPLE_CLASSES ${MODE} | 'base' for image classification with seen classes, 'new' for image classification with unseen classes, and 'unseen' for IRT and VQA. |
| TEST.EVALUATOR ${EVALUATOR} | 'Classification' for image classification, 'Retrieval' for ITR, and 'VQA' for VQA. |

An example for evaluating CoCoOp on OxfordPets with unseen classes:
```sh
cd CPL
python train.py \
    --root /path/to/datasets \
    --seed 1 \
    --trainer CoCoOp \
    --dataset-config-file configs/datasets/oxford_pets.yaml \
    --config-file configs/trainers/CoCoOp/vit_b16_c4_ep10_batch1_ctxv1.yaml \
    --model-dir /path/to/cocoop/model/output \
    --output-dir /path/to/cocoop/evaluation/output \
    --load-epoch 10 \
    --eval-only \
    DATASET.NUM_SHOTS 16 \
    DATASET.SUBSAMPLE_CLASSES new \
    TEST.EVALUATOR Classification
```
An example for training CPL on OxfordPets:
```sh
cd CPL
python train_cf.py \
    --root /path/to/datasets \
    --seed 1 \
    --trainer CoCoOpcf \
    --dataset-config-file configs/datasets/oxford_pets.yaml \
    --config-file configs/trainers/CoCoOp/vit_b16_c4_ep10_batch1_ctxv1.yaml \
    --model-dir /path/to/cpl/model/output \
    --output-dir /path/to/cpl/evaluation/output \
    --load-epoch 10 \
    --eval-only \
    DATASET.NUM_SHOTS 16 \
    DATASET.SUBSAMPLE_CLASSES new \
    TEST.EVALUATOR Classification
```

### Bash scripts
Bash scripts used to run the reproducibility experiments are included in `CPL/scripts`.

## Reproducibility results
The results of our reproducibility report are included in Jupyter notebooks in `CPL/notebooks`.


## Code structure
```
CPL/
├── Dassl.pytorch
│   ├── Dassl
│   │   ├── engine (modified trainer and dataset loader for CPL)
│   │   │   ├── ...
│   │   ├── evaluation (modified evaluator for ITR and VQA)
│   │   │   ├── ...
│   │   ├── metrics (mesaured metrics)
│   │   │   ├── ...
│   │   ├── ...
│   ├── ...
├── datasets (datatset processing files, added flickr8k, 30k, mscoco, and vqav2)
│   │   ├── flickr8k.py 
│   │   ├── flickr30k.py
│   │   ├── mscoco.py
│   │   ├── vqav2.py
├── trainers
│   ├── cocoop.py (original CoCoOp)
│   ├── cocoopcf.py (CPL)
│   ├── zsclip.py (zeroshot CLIP)
├── prompt (T5 generated prompts for VQAv2)
├── train_cf.py (main training file for CPL)
├── train.py (main training file for CoCoOp)
├──  ...
```


## Citation for the original paper
```
@inproceedings{he-2022-CPL,
    title = "{CPL}: Counterfactual Prompt Learning for Vision and Language Models",
    author = " Xuehai He and 
        Diji Yang and 
        Weixi Feng and
        Tsu-Jui Fu and
        Arjun Akula and 
        Varun Jampani and
        Pradyumna Narayana and 
        Sugato Basu and
        William Yang Wang and 
        Xin Eric Wang",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    year = "2022",
}
```
