#!/bin/bash 
set -e 
set -x 

python exec.py --config_path ./configs/finetune-full-train.jsonnet --batch_size 1
