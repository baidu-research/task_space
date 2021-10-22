#!/bin/bash

# exclude first pick
first_ckpt=$(awk 'NR==1' ../MMI_K5.txt);
exclude_id=$(grep -n "^$first_ckpt" ../random_K1.txt | cut -d':' -f1);
exclude_id=$((exclude_id-1)); # python id starts from 0

ckpts_to_watch="bert-base-uncased bert-large-cased"
watch_id=""
for ckpt in $ckpts_to_watch
do
  i=$(grep -n "^${ckpt}" ../random_K1.txt | cut -d':' -f1);
  i=$((i-1)); # python id starts from 0
  watch_id="$watch_id $i";
done
python violinplot.py task_logs/chunking/MMI task_logs/chunking/random -m f1 -e $exclude_id -si $watch_id -sm $ckpts_to_watch -y 0.7 0.97 -s fig3/chunking.eps
python violinplot.py task_logs/pos_ptb/MMI task_logs/pos_ptb/random -e $exclude_id -si $watch_id -sm $ckpts_to_watch -y 0.9 0.98 --no-legend -s fig3/pos_ptb.eps
python violinplot.py task_logs/ner/MMI task_logs/ner/random -m f1 -e $exclude_id -si $watch_id -sm $ckpts_to_watch -y 0.6 0.9 --no-legend -s fig3/ner.eps
python violinplot.py task_logs/ef/MMI task_logs/ef/random -m pearson -e $exclude_id -si $watch_id -sm $ckpts_to_watch -y 0.6 0.8 --no-legend -s fig3/ef.eps
python violinplot.py task_logs/st/MMI task_logs/st/random -e $exclude_id -si $watch_id -sm $ckpts_to_watch -y 0.86 0.95 --no-legend -s fig3/st.eps
python violinplot.py task_logs/syn_p/MMI task_logs/syn_p/random -e $exclude_id -si $watch_id -sm $ckpts_to_watch -y 0.85 0.97 --no-legend -s fig3/syn-p.eps
python violinplot.py task_logs/syn_gp/MMI task_logs/syn_gp/random -e $exclude_id -si $watch_id -sm $ckpts_to_watch --no-legend -s fig3/syn-gp.eps
python violinplot.py task_logs/syn_ggp/MMI task_logs/syn_ggp/random -e $exclude_id -si $watch_id -sm $ckpts_to_watch --no-legend -s fig3/syn-ggp.eps
