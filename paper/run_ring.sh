#!/bin/bash

export LD_LIBRARY_PATH=/Library/gurobi902/mac64/lib/
dir=$(pwd)
parentdir="$(dirname "$(pwd)")"
for model_name in $1 
do
    if [ $model_name = "transformer" ]; then 
        profile_dir="examples/transformer_proc/transformer_big_float32_hs_384"
    elif [ $model_name = "resnet_v1_50" ]; then 
        profile_dir="examples/resnet_v1_50_proc/resnet_v1_50_float32"
    elif [ $model_name = "gpt2_355M" ]; then 
        profile_dir="examples/gpt2_355M_proc/gpt2_355M_float32"
    else
        printf '%s\n' "model name not implemented." >&2
        exit 1
    fi
    profile_dir=$parentdir/$profile_dir
    max_dist=16
    for dec_interval_micro in 100
        do
    if [ $model_name = "gpt2_355M_proc" ]; then
        num_prof_flag="--num_profiles 5"
    else
        num_prof_flag="--num_profiles 10"
    fi

    for num_waves in 20 40 80 160 320 
    do
        for num_gpus in 1024 
        do
            screen_name="sipml-ring_${model_name}_${num_waves}waves"
            echo $screen_name
            log_dir=$dir/logs/$model_name/$strategy/ring/ng$num_gpus/num_waves$num_waves/dec_interval_micro$dec_interval_micro/
            cmnd="sipml-ring --num_gpus $num_gpus --num_waves $num_waves --max_dist $max_dist --dec_interval_micro $dec_interval_micro --bw_decision_type ILP $num_prof_flag --input_profile $profile_dir --log_dir $log_dir $2"
            echo $cmnd
            screen -dmS $screen_name bash -c "$cmnd"
        done
    done
    done
done
