#!/bin/bash
log=$1
grep "Loss:" $log | awk '{print $NF}' | gnuplot -p -e "plot '-' w l title 'train'" 
grep "Average loss:" $log | awk '{print $7}' | cut -d',' -f1 | gnuplot -p -e "plot '-' w l title 'test'"
grep "Accuracy:" $log | awk '{print $NF*100}' | gnuplot -p -e "plot '-' w l title 'accuracy (%)'"

