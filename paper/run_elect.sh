#!/bin/bash

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
    if [ $model_name = "gpt2_355M" ]; then
        num_prof_flag="--num_profiles 5"
    else
        num_prof_flag="--num_profiles 10"
    fi
profile_dir=$parentdir/$profile_dir
latency_us=0
for bw_per_port_Gb in 128 256 512 1024 2048 4096 8192
do
for num_gpus in 1024
do
    screen_name="sipml-elect_${model_name}_${bw_per_port_Gb}Gbps"
    echo $screen_name
    port_count=$num_gpus
    log_dir=$dir/logs/$model_name/$strategy/elect/ng$num_gpus/np$port_count/bw$bw_per_port_Gb/latency$latency_us/
    cmnd="sipml-elect --num_gpus $num_gpus --bw_per_port_Gb $bw_per_port_Gb --latency_us $latency_us --input_profile $profile_dir $num_prof_flag --log_dir $log_dir $2"
    echo $cmnd
    screen -dmS $screen_name bash -c "$cmnd"
done
done
done
