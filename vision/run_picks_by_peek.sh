#!/bin/bash
ids=$1
kk=$2
tasks=$3
exp_root=checkpoint
exp_dir=${exp_root}/${ids}
determine_by_task=0
total_agents=$(ls ${exp_dir}/agents | wc -l)
partition=your_GPU_partition


for a in `seq 0 $((total_agents-1))`
do
  printf -v aa "%02d" $a;
  sbatch -p $partition --gres=gpu:1 --wrap "python admin_trains_classification_head.py ${exp_dir}/task_features/${aa}.npz --task-id $determine_by_task -e 1500" -o ${exp_dir}/peek_logs/use_${aa}.out;
done

for a in `seq 0 $((total_agents-1))`
do
  printf -v aa "%02d" $a;
  tail -n1 ${exp_dir}/peek_logs/use_${aa}.out | cut -d',' -f1 | awk -v ag=${aa} '{print ag, $NF}'
done | sort -k2 -nr | head -n $kk > ${exp_dir}/peek_picks

rm ${exp_dir}/peek_logs/use_*.out;

files=""
for k in `seq 1 ${kk}`;
do
  k_th=$(cut -d' ' -f1 ${exp_dir}/peek_picks | sed "${k}p;d")
  files+=" ${exp_dir}/task_features/${k_th}.npz"
  for t in `seq 0 $((tasks-1))`;
  do
    sbatch -p $partition --gres=gpu:1 --wrap "python admin_trains_classification_head.py ${files} --task-id $t -e 1500" -o ${exp_dir}/peek_logs/peek_k${k}_t${t}.out;
  done
done
