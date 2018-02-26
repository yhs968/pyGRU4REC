import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from modules.model_old import GRU4REC
import torch


def main():
    
    # parse the nn arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', default=100, type=int)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--dropout_input', default=0, type=float)
    parser.add_argument('--dropout_hidden', default=.5, type=float)

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
    parser.add_argument('--n_samples', default=-1, type=int)
    args = parser.parse_args()
    
    # Get the arguments
    
    PATH_HOME = Path.home()
    PATH_PROJ = PATH_HOME/'pyGRU4REC' 
    PATH_DATA = PATH_PROJ/'data'
    train = 'rsc15_train_full.txt'
    test = 'rsc15_test.txt'
    PATH_TO_TRAIN = PATH_DATA / train
    PATH_TO_TEST = PATH_DATA / test

    df_train = pd.read_csv(PATH_TO_TRAIN, sep='\t', dtype={'ItemId': np.int64})
    df_test = pd.read_csv(PATH_TO_TEST, sep='\t', dtype={'ItemId': np.int64})

    # sampling, if needed
    n_samples = args.n_samples
    if n_samples != -1:
        df_train = df_train[:n_samples]
        df_test = df_test[:n_samples]
    
    session_key = 'SessionId'
    time_key = 'Time'
    item_key = 'ItemId'

    use_cuda = True
    input_size = df_train[item_key].nunique()
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    output_size = input_size
    batch_size = args.batch_size
    dropout_input = args.dropout_input
    dropout_hidden = args.dropout_hidden
    
    loss_type = args.loss_type
    
    optimizer_type = args.optimizer_type
    lr = args.lr
    weight_decay = args.weight_decay
    momentum = args.momentum
    eps = args.eps
   
    n_epochs = args.n_epochs
    time_sort = args.time_sort

    torch.manual_seed(7)
    torch.cuda.manual_seed(7)

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
                    dropout_input=dropout_input,
                    dropout_hidden=dropout_hidden,
                    time_sort=time_sort)

    model.train(df_train, session_key, time_key, item_key, n_epochs=n_epochs)


if __name__ == '__main__':
    main()
