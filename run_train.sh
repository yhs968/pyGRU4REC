CUDA_VISIBLE_DEVICES=2 nohup python run_train.py --loss_type BPR --optimizer_type Adagrad --dropout_hidden 0.2 --lr 0.05 --momentum 0.2 > log2.out &
CUDA_VISIBLE_DEVICES=5 nohup python run_train.py --loss_type BPR --optimizer_type Adam --dropout_hidden 0.2 --lr 0.05 --momentum 0.2 > log5.out &
