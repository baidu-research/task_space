#!/bin/bash

log_file=$1
acc=$(tail -n1 $log_file | cut -d',' -f1 | awk '{print $NF}')
echo $acc
