#!/bin/bash

get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}

ids=$1 # e.g., N30_T20_S10_P10_K100_C30 
seed=$2
kk=$3 # max number of agents
tasks=$4 # number of admin tasks
agents=$5 # total numner of agents
exp_root=checkpoint
exp_dir=${exp_root}/${ids}
partition=your_GPU_partition


for k in `seq 1 ${kk}`;
do
  takes=$(shuf --random-source=<(get_seeded_random $((seed+k))) -i 0-$((agents-1)) -n $k);
  files=""
  for id in $takes;
  do
    printf -v id_formated "%03d" $id
    files+=" ${exp_dir}/task_features/${id_formated}.npz"
  done
  for t in `seq 0 $((tasks-1))`;
  do
    sbatch -p $partition --gres=gpu:1 --wrap "python admin_trains_classification_head.py ${files} --task-id $t -e 1500" -o ${exp_dir}/random_logs/seed${seed}_k${k}_t${t}.out;
  done
done
