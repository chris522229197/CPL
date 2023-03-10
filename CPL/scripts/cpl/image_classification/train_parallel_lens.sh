#!/bin/bash

dataset=${1}
mode=${2}
device1=${3}
device2=${4}
device3=${5}
device4=${6}
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
    train_command+=" --config-file configs/trainers/CoCoOp/vit_b16_c4_ep10_batch1_ctx_len${len}.yaml"

    output_dir="/projects/leelab/clin25/cpl-output-len${len}"
    mkdir "${output_dir}"
    output_dir+="/${dataset}"
    mkdir "${output_dir}"
    output_dir+="/${mode}"
    mkdir "${output_dir}"
    output_dir+="/${seed}"
    mkdir "${output_dir}"

    train_command+=" --output-dir ${output_dir}"
    train_command+=" DATASET.NUM_SHOTS 16"
    train_command+=" DATASET.SUBSAMPLE_CLASSES ${mode}"

    screen_name+="-${dataset}-${mode}-len${len}-${seed}-train"
    screen -dmS "${screen_name}" bash -c "${train_command}"
done
