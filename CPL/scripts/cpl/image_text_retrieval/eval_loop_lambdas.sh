#!/bin/bash

dataset=${1}
num_shots=${2}
device=${3}
lambdas=( "0" "5" "10" )
seed="1"
eval_mode="unseen"

for i in {0..2}
do
    lambda="${lambdas[i]}"
    model_dir="/projects/leelab/clin25/cpl-output-lambda${lambda}/${dataset}/${num_shots}-shots/${seed}"
    eval_dir="${model_dir}/eval-${eval_mode}"
    mkdir "${eval_dir}"

    CUDA_VISIBLE_DEVICES=${device}, \
    python train_cf.py \
    --root /projects/leelab/data/image \
    --seed "${seed}" \
    --trainer CoCoOpcf \
    --dataset-config-file "configs/datasets/${dataset}.yaml" \
    --config-file "configs/trainers/cpl/vit_b16_c4_ep10_batch1_ctxv1_lambda${lambda}.yaml" \
    --model-dir "${model_dir}" \
    --output-dir "${eval_dir}" \
    --load-epoch 10 \
    --eval-only \
    DATASET.NUM_SHOTS "${num_shots}" \
    DATASET.SUBSAMPLE_CLASSES "${eval_mode}" \
    TEST.EVALUATOR Retrieval
done
