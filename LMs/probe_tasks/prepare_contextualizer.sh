#!/bin/bash
# For a specified checkpoint, extract features for all tasks

OUTPUT_DIR=contextualizers
ckpt=$1 # see ../ckpts.txt for a list of huggingface checkpoints 
mkdir -p ${OUTPUT_DIR}/${ckpt}

while IFS='' read -r line || [[ -n "$line" ]]; do
    line=${line//$' '/,}
    IFS=',' read -r -a columns <<< "$line"
    task_name="${columns[0]%.*}"
    sbatch -p 1080Ti_short --gres=gpu:1 --wrap "python prepare_contextualizer.py ${ckpt} ${columns[1]} -s ${OUTPUT_DIR}/${ckpt}/${columns[0]}" 
done < "task_sentences.txt"
