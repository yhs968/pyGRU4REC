from pathlib import Path
import pandas as pd
import numpy as np
from modules.layer import GRU
from modules.model import GRU4REC
import torch
import argparse

def main():
    parser = argparse.ArgumentParser()
    
    # Model filename
    parser.add_argument('model_file', type=str)
    
    # Size of the recommendation list
    parser.add_argument('--k', default=20, type=int)
    
    # parse the nn arguments
    
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
    parser.add_argument('--n_epochs', default=2, type=int)
    parser.add_argument('--time_sort', default=False, type=bool)
    parser.add_argument('--n_samples', default=-1, type=int)
    
    # Get the arguments
    args = parser.parse_args()

    PATH_DATA = Path('./data')
    PATH_MODEL = Path('./models')
    train = 'train.tsv'
    test = 'test.tsv'
    PATH_TRAIN = PATH_DATA / train
    PATH_TEST = PATH_DATA / test
    

    df_train = pd.read_csv(PATH_TRAIN, sep='\t', names=['SessionId','ItemId','TimeStamp'])
    df_test = pd.read_csv(PATH_TEST, sep='\t', names=['SessionId','ItemId','TimeStamp'])

    # sampling, if needed
    n_samples = args.n_samples
    if n_samples != -1:
        df_train = df_train[:n_samples]
        df_test = df_test[:n_samples]
        
    session_key = 'SessionId'
    item_key = 'ItemId'
    time_key = 'TimeStamp'
    
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
    
    MODEL_FILE = PATH_MODEL/args.model_file
    
    gru = GRU(input_size, hidden_size, output_size,
          num_layers = num_layers,
          dropout_input = dropout_input,
          dropout_hidden = dropout_hidden,
          batch_size = batch_size,
          use_cuda = use_cuda)

    gru.load_state_dict(torch.load(MODEL_FILE))

    model = GRU4REC(input_size, hidden_size, output_size,
                    num_layers = num_layers,
                    dropout_input = dropout_input,
                    dropout_hidden = dropout_hidden,
                    batch_size = batch_size,
                    use_cuda = use_cuda,
                    loss_type = loss_type,
                    optimizer_type = optimizer_type,
                    lr=lr,
                    momentum=momentum,
                    time_sort=time_sort,
                    pretrained=gru)

    model.init_data(df_train, df_test, session_key=session_key, time_key=time_key, item_key=item_key)

    k = args.k
    recall, mrr = model.test(k=k, batch_size=batch_size)
    result = f'Recall@{k}:{recall:.7f},MRR@{k}:{mrr:.7f}'
    print(result)

if __name__ == '__main__':
    main()
