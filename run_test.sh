#!/usr/bin/env bash
nohup python run_test.py GRU4REC_TOP1_Adagrad_0.01_epoch10 --loss_type TOP1 --optimizer_type Adagrad --dropout_hidden 0.5 --lr 0.01 --momentum 0 --batch_size 50 > logs/test_top1.out &
nohup python run_test.py GRU4REC_BPR_Adagrad_0.05_epoch10 --loss_type BPR --optimizer_type Adagrad --dropout_hidden 0.2 --lr 0.05 --momentum 0.2 --batch_size 50 > logs/test_bpr.out &
nohup python run_test.py GRU4REC_CrossEntropy_Adagrad_0.01_epoch10 --loss_type CrossEntropy --optimizer_type Adagrad --dropout_hidden 0 --lr 0.01 --momentum 0 --batch_size 500 > logs/test_xe.out &