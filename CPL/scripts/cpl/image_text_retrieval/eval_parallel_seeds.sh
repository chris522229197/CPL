#!/bin/bash

dataset=${1}
num_shots=${2}
device1=${3}
device2=${4}
device3=${5}
devices=( "${device1}" "${device2}" "${device3}" )

for i in {0..2}
do
    device="${devices[i]}"
    if [ "${device}" = "-1" ]
    then
        continue
    fi

    seed=$((i + 1))
    eval_command=""
    screen_name="cpl"

    eval_command+="source /homes/gws/clin25/miniconda3/etc/profile.d/conda.sh;"
    eval_command+=" conda activate dassl-cpl;"
    eval_command+=" CUDA_VISIBLE_DEVICES=${device},"
    eval_command+=" python train_cf.py"

    eval_command+=" --root /projects/leelab/data/image"
    eval_command+=" --seed ${seed}"
    eval_command+=" --trainer CoCoOpcf"
    eval_command+=" --dataset-config-file configs/datasets/${dataset}.yaml"
    eval_command+=" --config-file configs/trainers/CoCoOp/vit_b16_c4_ep10_batch1_ctxv1.yaml"

    few_shot_model_dir="${dataset}/${num_shots}-shots/${seed}"
    model_dir="/projects/leelab/clin25/cpl-output/${few_shot_model_dir}"
    eval_output_dir="${model_dir}/eval-unseen"
    mkdir "${eval_output_dir}"

    eval_command+=" --model-dir ${model_dir}"
    eval_command+=" --output-dir ${eval_output_dir}"

    eval_command+=" --load-epoch 10"
    eval_command+=" --eval-only"
    eval_command+=" DATASET.NUM_SHOTS ${num_shots}"
    eval_command+=" DATASET.SUBSAMPLE_CLASSES unseen"

    screen_name+="-${dataset}-${num_shots}shots-${seed}-eval"
    screen -dmS "${screen_name}" bash -c "${eval_command}"
done
