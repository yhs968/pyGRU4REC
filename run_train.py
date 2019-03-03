from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import torch
from modules.model import GRU4REC
from modules.data import SessionDataset


def main():
    
    # parse the nn arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', default=100, type=int)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--p_dropout_input', default=0, type=float)
    parser.add_argument('--p_dropout_hidden', default=.5, type=float)

    # parse the optimizer arguments
    parser.add_argument('--optimizer_type', default='Adagrad', type=str)
    parser.add_argument('--lr', default=.01, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--momentum', default=0, type=float)
    parser.add_argument('--eps', default=1e-6, type=float)
    
    # parse the loss type
    parser.add_argument('--loss_type', default='TOP1', type=str)
    
    # etc
    parser.add_argument('--n_epochs', default=10, type=int)
    parser.add_argument('--time_sort', default=False, type=bool)
    parser.add_argument('--model_name', default='GRU4REC', type=str)
    
    # Get the arguments
    args = parser.parse_args()    

    PATH_DATA = Path('./data')
    PATH_MODEL = Path('./models')
    train = 'train.tsv'
    test = 'test.tsv'
    PATH_TRAIN = PATH_DATA / train
    PATH_TEST = PATH_DATA / test

    train_dataset = SessionDataset(PATH_TRAIN)
    test_dataset = SessionDataset(PATH_TEST, itemmap=train_dataset.itemmap)

    use_cuda = True
    input_size = len(train_dataset.items)
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    output_size = input_size
    batch_size = args.batch_size
    p_dropout_input = args.p_dropout_input
    p_dropout_hidden = args.p_dropout_hidden
    
    loss_type = args.loss_type
    
    optimizer_type = args.optimizer_type
    lr = args.lr
    weight_decay = args.weight_decay
    momentum = args.momentum
    eps = args.eps
   
    n_epochs = args.n_epochs
    time_sort = args.time_sort

    torch.manual_seed(7)

    model = GRU4REC(input_size, hidden_size, output_size,
                    num_layers=num_layers,
                    use_cuda=use_cuda,
                    batch_size=batch_size,
                    loss_type=loss_type,
                    optimizer_type=optimizer_type,
                    lr=lr,
                    weight_decay=weight_decay,
                    momentum=momentum,
                    eps=eps,
                    p_dropout_input=p_dropout_input,
                    p_dropout_hidden=p_dropout_hidden,
                    time_sort=time_sort)
    
    model.train(train_dataset, k=20, n_epochs=n_epochs, model_name=args.model_name, save=True, save_dir=PATH_MODEL)
    model.test(test_dataset, k=20)

if __name__ == '__main__':
    main()
