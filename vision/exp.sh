#!/bin/bash

# create split of admin and agent
N=30 # admin handles N-way classification
T=20 # admin has T tasks, each a N-way classification
S=10 # admin has S training samples per class
P=10 # reseve P*100 samples as probe samples
K=100 # K agents
C=30  # each agent trained on a C-way classification task
#non_overlap_c=50 # agents see these classes, admin sees the rest, TODO
exp_root=checkpoint
partition=your_GPU_partition

ids=N${N}_T${T}_S${S}_P${P}_K${K}_C${C};

exp_dir=${exp_root}/${ids}
mkdir -p $exp_dir
python create_data_split.py \
    --admin-classes $N \
    --admin-tasks $T \
    --admin-samples $S \
    --probe-samples $P \
    --agents $K \
    --agent-classes $C \
    --save ${exp_dir}/split.npz;

# train agents
mkdir ${exp_dir}/agents
for a in `seq 0 $((K-1))`;
do
  printf -v aa "%02d" $a;
  mkdir ${exp_dir}/agents/agent_${aa};
  sbatch -p $partition --gres=gpu:1 --wrap "python agents.py ${exp_dir}/split.npz ${a} -s ${ids}/agents/agent_${aa}" -o ${exp_dir}/agents/agent_${aa}/log;
done

# admin computes agents' similarities
mkdir ${exp_dir}/probe_features
for a in `seq 0 $((K-1))`;
do
  printf -v aa "%02d" $a;
  sbatch -p $partition --gres=gpu:1 --wrap "python admin_extract_probe_features.py ${exp_dir}/agents/agent_${aa} ${exp_dir}/split.npz ${exp_dir}/probe_features/${aa}.npy -r -5 5 -10 10 -15 15" -o extract_${aa}.log;
done
python admin_computes_agent_sim.py ${exp_dir}/probe_features/*.npy --save ${exp_dir}/kappa

# admin extracts features on task data, based on each agent checkpoint
mkdir ${exp_dir}/task_features
for a in `seq 0 $((K-1))`;
do
  printf -v aa "%02d" $a;
  sbatch -p $partition --gres=gpu:1 --wrap "python admin_prepare_features_for_new_tasks.py ${exp_dir}/agents/agent_${aa} ${exp_dir}/split.npz ${exp_dir}/task_features/${aa}.npz" -o extract_${aa}.log;
done
rm extract_*.log;

# max number of agents to pick
kk=20

# random pick baseline
mkdir ${exp_dir}/random_logs
for seed in `seq 0 10 90`;
do
  ./run_picks_by_random.sh $ids $seed $kk $T $K;
done

# top-k on one (actually first) admin task. It excels on that task, but fails on the other ones
mkdir ${exp_dir}/peek_logs
./run_picks_by_topk.sh $ids $kk $T

# MMI picks
python ../MMI.py ${exp_dir}/kappa.npy -k $kk --hub-ini > ${exp_dir}/MI_picks;
mkdir ${exp_dir}/MI_logs
./run_picks_by_MI.sh ${ids} ${exp_dir}/MI_picks $kk $T;

python analyze_result.py ${exp_dir}/MI_logs/ ${exp_dir}/peek_logs/ ${exp_dir}/random_logs/ -l MMI peek random
