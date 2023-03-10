#!/bin/bash

dataset=${1}
num_shots=${2}
device=${3}
lambdas=( "0" "5" "10" )
seed="1"

for i in {0..2}
do
    lambda="${lambdas[i]}"
    out_dir="/projects/leelab/clin25/cpl-output-lambda${lambda}"
    mkdir "${out_dir}"
    out_dir+="/${dataset}"
    mkdir "${out_dir}"
    out_dir+="/${num_shots}-shots"
    mkdir "${out_dir}"
    out_dir+="/${seed}"
    mkdir "${out_dir}"

    CUDA_VISIBLE_DEVICES=${device}, \
    python train_cf.py \
    --root /projects/leelab/data/image \
    --seed "${seed}" \
    --trainer CoCoOpcf \
    --dataset-config-file "configs/datasets/${dataset}.yaml" \
    --config-file "configs/trainers/cpl/vit_b16_c4_ep10_batch1_ctxv1_lambda${lambda}.yaml" \
    --output-dir "${out_dir}" \
    DATASET.NUM_SHOTS "${num_shots}" \
    DATASET.SUBSAMPLE_CLASSES fewshot
done
