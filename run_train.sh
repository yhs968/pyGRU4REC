#!/bin/bash
nohup python -u run_train.py --loss_type TOP1 --optimizer_type Adagrad --p_dropout_hidden 0.5 --lr 0.01 --momentum 0 --batch_size 50 --n_epochs 20 > logs/train_top1.out &
nohup python -u run_train.py --loss_type BPR --optimizer_type Adagrad --p_dropout_hidden 0.2 --lr 0.05 --momentum 0.2 --batch_size 50 --n_epochs 10 > logs/train_bpr.out &
nohup python -u run_train.py --loss_type CrossEntropy --optimizer_type Adagrad --p_dropout_hidden 0 --lr 0.01 --momentum 0 --batch_size 500 --n_epochs 10 > logs/train_xe.out &
