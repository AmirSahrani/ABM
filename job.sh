#!/bin/bash

# Loop through each file in the configs directory
for conf in ./configs/center* ./configs/random*
do
  echo $conf
  echo "starting"
  python "scripts/experiment.py" "$conf"
  echo "finished"
done
