#!/bin/bash
set -e 
set -x
#python eval.py --archive_path model.tar.gz --dev_path "/home/ashutosh/HDD/IITB/Sem_1/MS_RnD/ms_rnd1/generate_templated/spider_format_train.json" --output custom_preds_train.sql --gpu 0

python smbop/eval_final/evaluation.py --gold "/home/ashutosh/HDD/IITB/Sem_1/MS_RnD/ms_rnd1/generate_templated/spider_gold_train.sql" --pred custom_preds_train.sql --etype all --db  dataset/database  --table dataset/tables.json #> output_eval_custom.txt
