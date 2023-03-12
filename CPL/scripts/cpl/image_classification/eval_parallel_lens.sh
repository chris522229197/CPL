#!/bin/bash

dataset=${1}
mode=${2}
eval_mode=${3}
device1=${4}
device2=${5}
device3=${6}
device4=${7}
devices=( "${device1}" "${device2}" "${device3}" "${device4}" )
lens=( "8" "16" "32" "64" )
seed=1

for i in {0..3}
do
    device="${devices[i]}"
    if [ "${device}" = "-1" ]
    then
        continue
    fi

    len="${lens[i]}"

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
    eval_command+=" --config-file configs/trainers/CoCoOp/vit_b16_c4_ep10_batch1_ctx_len${len}.yaml"

    model_dir="/projects/leelab/clin25/cpl-output-len${len}/${dataset}/${mode}/${seed}"
    eval_output_dir="${model_dir}/eval-${eval_mode}"
    mkdir "${eval_output_dir}"

    eval_command+=" --model-dir ${model_dir}"
    eval_command+=" --output-dir ${eval_output_dir}"

    eval_command+=" --load-epoch 10"
    eval_command+=" --eval-only"
    eval_command+=" DATASET.NUM_SHOTS 16"
    eval_command+=" DATASET.SUBSAMPLE_CLASSES ${eval_mode}"

    screen_name+="-${dataset}-${eval_mode}-len${len}-${seed}-eval"
    screen -dmS "${screen_name}" bash -c "${eval_command}"
done
