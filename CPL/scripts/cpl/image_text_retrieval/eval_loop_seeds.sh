#!/bin/bash

dataset=${1}
num_shots=${2}
device=${3}

for i in {0..2}
do
    seed=$((i + 1))
    few_shot_model_dir="${dataset}/${num_shots}-shots/${seed}"
    model_dir="/projects/leelab/clin25/cpl-output/${few_shot_model_dir}"
    eval_output_dir="${model_dir}/eval-unseen"
    mkdir "${eval_output_dir}"

    CUDA_VISIBLE_DEVICES=${device}, \
    python train_cf.py \
    --root /projects/leelab/data/image \
    --seed "${seed}" \
    --trainer CoCoOpcf \
    --dataset-config-file "configs/datasets/${dataset}.yaml" \
    --config-file configs/trainers/CoCoOp/vit_b16_c4_ep10_batch1_ctxv1.yaml \
    --model-dir "${model_dir}" \
    --output-dir "${eval_output_dir}" \
    --load-epoch 10 \
    --eval-only \
    DATASET.NUM_SHOTS "${num_shots}" \
    DATASET.SUBSAMPLE_CLASSES unseen \
    TEST.EVALUATOR Retrieval
done
