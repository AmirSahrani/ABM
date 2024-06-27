#!/bin/bash

# Loop through each file in the configs directory
for conf in ./configs/sandor2*
do
  echo $conf
  echo "starting"
  python "scripts/experiment.py" "$conf"
  echo "finished"
done
