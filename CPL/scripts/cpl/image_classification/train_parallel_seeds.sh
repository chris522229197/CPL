#!/bin/bash

dataset=${1}
mode=${2}
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
    train_command=""
    screen_name="cpl"

    train_command+="source /homes/gws/clin25/miniconda3/etc/profile.d/conda.sh;"
    train_command+=" conda activate dassl-cpl;"
    train_command+=" CUDA_VISIBLE_DEVICES=${device},"
    train_command+=" python train_cf.py"
    train_command+=" --root /projects/leelab/data/image"
    train_command+=" --seed ${seed}"
    train_command+=" --trainer CoCoOpcf"
    train_command+=" --dataset-config-file configs/datasets/${dataset}.yaml"
    train_command+=" --config-file configs/trainers/CoCoOp/vit_b16_c4_ep10_batch1_ctxv1.yaml"
    train_command+=" --output-dir /projects/leelab/clin25/cpl-output/${dataset}/${mode}/${seed}"
    train_command+=" DATASET.NUM_SHOTS 16"
    train_command+=" DATASET.SUBSAMPLE_CLASSES ${mode}"

    screen_name+="-${dataset}-${mode}-${seed}-train"

    screen -dmS "${screen_name}" bash -c "${train_command}"
done
