#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0
#nohup python run_train.py --loss_type TOP1 --optimizer_type Adagrad --dropout_hidden 0.5 --lr 0.01 --momentum 0 --n_samples 1000 > logs/top1_1000.out &&
#nohup python run_train.py --loss_type BPR --optimizer_type Adagrad --dropout_hidden 0.2 --lr 0.05 --momentum 0.2 --n_samples 1000 > logs/bpr_1000.out &&
#nohup python run_train.py --loss_type CrossEntropy --optimizer_type Adagrad --dropout_hidden 0 --lr 0.01 --momentum 0 --batch_size 50 --n_samples 1000 > logs/xe_1000.out &
#nohup python run_train_simple.py --loss_type TOP1 --optimizer_type Adagrad --dropout_hidden 0.5 --lr 0.01 --momentum 0 --n_samples 1000 > logs/top1_1000_simple.out &&
#nohup python run_train_simple.py --loss_type BPR --optimizer_type Adagrad --dropout_hidden 0.2 --lr 0.05 --momentum 0.2 --n_samples 1000 > logs/bpr_1000_simple.out &&
#nohup python run_train_simple.py --loss_type CrossEntropy --optimizer_type Adagrad --dropout_hidden 0 --lr 0.01 --momentum 0 --batch_size 50 --n_samples 1000 > logs/xe_1000_simple.out &
nohup python run_train.py --loss_type TOP1 --optimizer_type Adagrad --dropout_hidden 0.5 --lr 0.01 --momentum 0 --batch_size 50 > logs/top1.out &
nohup python run_train.py --loss_type BPR --optimizer_type Adagrad --dropout_hidden 0.2 --lr 0.05 --momentum 0.2 --batch_size 50 > logs/bpr.out &
nohup python run_train.py --loss_type CrossEntropy --optimizer_type Adagrad --dropout_hidden 0 --lr 0.01 --momentum 0 --batch_size 500 > logs/xe.out &