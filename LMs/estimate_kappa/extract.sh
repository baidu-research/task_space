#!/bin/bash
data=$1 # text/wiki.train.len_noLess_10.tokens, or text/billion_10K.txt
save_path=$2

# read each ckpt from huggingface, and extract word representations
while read ckpt;
do
  python extract_feature.py $ckpt $data -b 4 -s ${save_path}/${ckpt} --save-at 32 128 512 1024 4096;
done < ../ckpts.txt
