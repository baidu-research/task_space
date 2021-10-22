#!/bin/bash
ids=$1 # e.g., N30_T20_S10_P10_K100_C30
pick_file=$2
kk=$3 # max number of agents to pick
tasks=$4 # number of admin tasks
exp_root=checkpoint
exp_dir=${exp_root}/${ids}
partition=your_GPU_partition

for k in `seq 1 ${kk}`;
do
  takes=$(cut -d' ' -f-$k ${pick_file})
  files=""
  for id in $takes
  do
    printf -v id_formated "%03d" $id
    files+=" ${exp_dir}/task_features/${id_formated}.npz"
  done
  for t in `seq 0 $((tasks-1))`;
  do
    log_name=MI_k${k}_t${t}.out
    sbatch -p $partition --gres=gpu:1 --wrap "python admin_trains_classification_head.py ${files} --task-id $t -e 2000" -o ${exp_dir}/MI_logs/$log_name;
  done
done
