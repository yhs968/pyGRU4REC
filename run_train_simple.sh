#!/usr/bin/env bash
nohup python run_train_simple.py --loss_type TOP1 --optimizer_type Adagrad --dropout_hidden 0.5 --lr 0.01 --momentum 0 --batch_size 50 > logs/top1.out &