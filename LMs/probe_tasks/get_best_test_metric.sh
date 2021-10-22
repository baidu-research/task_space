#!/bin/bash

log_file=$1
metric=${2:-acc_k=1}  # metric name
best_valid=$(grep '^dev' $log_file | awk -F'|' -v field=$metric 'BEGIN{best=0} {for(i=1; i<=NF; i++){split($i, arr, " "); if(arr[1]==field && arr[2]>best) {best=arr[2]; best_epoch=NR}}} END{print best_epoch, best}')
best_epoch=$(echo $best_valid | cut -d' ' -f1)
best_valid_metric=$(echo $best_valid | cut -d' ' -f2)
best_test_metric=$(grep '^test' $log_file | awk -F'|' -v e=$best_epoch -v field=$metric 'NR==e {for(i=1; i<=NF; i++){split($i, arr, " "); if(arr[1]==field) print arr[2]}}')
echo $((best_epoch-1)) $best_valid_metric $best_test_metric
