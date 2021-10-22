#!/bin/bash

$partition=your_GPU_partition
mkdir -p task_logs/syn_gp/random
mkdir -p task_logs/syn_gp/MMI

# MMI
for k in `seq 1 5`;
do
  ckpts=$(awk -v k=$k 'NR<=k {printf("%s ", $0)}' ../MMI_K5.txt);
  sbatch -p $partition --gres=gpu:1 --wrap "python train.py syn_gp -lms ${ckpts} -lr 1e-3 -e 40" -o task_logs/syn_gp/MMI/k${k}.log;
done

# random
for k in `seq 1 5`;
do
  s=0;
  while read -r ckpts;
  do
    sbatch -p $partition --gres=gpu:1 --wrap "python train.py syn_gp -lms ${ckpts} -lr 1e-3 -e 40" -o task_logs/syn_gp/random/k${k}_s${s}.log;
    s=$((s+1));
  done < ../random_K${k}.txt
done
