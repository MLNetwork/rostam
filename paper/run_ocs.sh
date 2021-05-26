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
    for reconf_delay_micro in 20 #20 100 300 1000 3000
        do
            dec_interval_micro=$(expr $reconf_delay_micro \* 5)
    if [ ["$2" == *"--single_shot"*] ]; then
        single_shot="single_shot"
    else
        single_shot=dec_interval_micro$dec_interval_micro/reconf_delay_micro$reconf_delay_micro/
    fi

    if [ $model_name = "gpt2_355M" ]; then
        num_prof_flag="--num_profiles 5"
    else
        num_prof_flag="--num_profiles 10"
    fi

    for num_ocs in 20 #9 10 12 14 15 16 
    do
        for num_waves in 20 40 80 160 320 
        do
            for num_gpus in 1024 
            do
                screen_name="sipml-ocs_${model_name}_${num_waves}waves"
                echo $screen_name
                port_count=$num_gpus
                log_dir=$dir/logs/$model_name/$strategy/ocs/ng$num_gpus/num_ocs$num_ocs/num_waves$num_waves/$single_shot/
                cmnd="sipml-ocs --num_gpus $num_gpus --num_waves $num_waves --num_ocs $num_ocs --port_count $port_count --dec_interval_micro $dec_interval_micro --reconf_delay_micro $reconf_delay_micro --input_profile $profile_dir $num_prof_flag --log_dir $log_dir $2"
                echo $cmnd
                screen -dmS $screen_name bash -c "$cmnd"
            done
        done
    done
    #done
    done
done
