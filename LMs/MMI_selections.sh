#!/bin/bash

for i in $(python ../MMI.py estimate_kappa/kappa.wiki2.all.npy -k 5);
do
  sed "$((i+1))q;d" ckpts.txt;
done > MMI_K5.txt
