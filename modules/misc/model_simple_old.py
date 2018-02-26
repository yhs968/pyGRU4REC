import time
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from modules.optimizer import Optimizer
from modules.loss import LossFunction
from modules.layer import GRU
import modules.evaluate as E
import modules.misc.generator as G


class GRU4REC:
    def __init__(self, input_size, hidden_size, output_size, num_layers=1,
                 optimizer_type='Adagrad', lr=.01, weight_decay=0,
                 momentum=0, eps=1e-6, loss_type='TOP1',
                 clip_grad=-1, dropout_input=.0, dropout_hidden=.5,
                 batch_size=50, use_cuda=True, time_sort=False, pretrained=None):
        '''
        Args:
            input_size (int): dimension of the gru input variables
            hidden_size (int): dimension of the gru hidden units
            output_size (int): dimension of the gru output variables
            optimizer_type (str): optimizer type for GRU weights
            lr (float): learning rate for the optimizer
            weight_decay (float): weight decay for the optimizer
            momentum (float): momentum for the optimizer
            eps (float): eps for the optimizer
            loss_type (str): type of the loss function to use
            clip_grad (float): clip the gradient norm at clip_grad. No clipping if clip_grad = -1
            dropout_input (float): dropout probability for the input layer
            dropout_hidden (float): dropout probability for the hidden layer
            batch_size (int): mini-batch size
            use_cuda (bool): whether you want to use cuda or not
            time_sort (bool): whether to ensure the the order of sessions is chronological (default: False)
            pretrained (modules.layer.GRU): pretrained GRU layer, if it exists (default: None)
        '''
        
        # Initialize the GRU Layer
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        if pretrained is None:
            self.gru = GRU(input_size, hidden_size, output_size, num_layers,
                           dropout_input=dropout_input,
                           dropout_hidden=dropout_hidden,
                           use_cuda=use_cuda,
                           batch_size=batch_size)
        else:
            self.gru = pretrained
        # Initialize the optimizer
        self.optimizer_type = optimizer_type
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lr = lr
        self.eps = eps
        self.optimizer = Optimizer(self.gru.parameters(),
                                   optimizer_type=optimizer_type,
                                   lr=lr,
                                   weight_decay=weight_decay,
                                   momentum=momentum,
                                   eps=eps)
        # Initialize the loss function
        self.loss_type = loss_type
        self.loss_fn = LossFunction(loss_type, use_cuda)
        # gradient clipping(optional)
        self.clip_grad = clip_grad 
        # etc
        self.time_sort = time_sort

    def train(self, n_epochs=10, save_dir='./models', model_name='GRU4REC'):
        # Time the training process
        start_time = time.time()
        for epoch in range(n_epochs):
            loss = self.run_epoch()
            end_time = time.time()
            wall_clock = (end_time - start_time) / 60
            print(f'Epoch:{epoch+1:2d}/Loss:{loss:0.3f}/TrainingTime:{wall_clock:0.3f}(min)')
            start_time = time.time()
            
            # Store the intermediate model
            save_dir = Path(save_dir)
            
            model_fname = f'{model_name}_{self.loss_type}_{self.optimizer_type}_{self.lr}_epoch{epoch+1:d}'
            torch.save(self.gru.state_dict(), save_dir/model_fname)

    def run_epoch(self):
        # initialize
        mb_losses = []
        hidden = self.gru.init_hidden().data
        optimizer = self.optimizer

        # Start the training loop
        for input, target in G.generate_batch(df=self.df_train,
                                              batch_size=self.batch_size,
                                              hidden=hidden,
                                              training=True,
                                              time_sort=self.time_sort):
            if self.use_cuda:
                input = input.cuda()
                target = target.cuda()
            # Embed the input
            embedded = self.gru.emb(input)
            # Go through the GRU layer
            logit, hidden = self.gru(embedded, target, hidden)
            # Calculate the mini-batch loss
            mb_loss = self.loss_fn(logit)
            mb_losses.append(mb_loss.data[0])
            # flush the gradient b/f backprop
            optimizer.zero_grad()
            # Backprop
            mb_loss.backward()
            # Gradient Clipping(Optional)
            if self.clip_grad != -1:
                for p in self.gru.parameters():
                    p.grad.data.clamp_(max=self.clip_grad)
            # Mini-batch GD
            optimizer.step()
            # Detach the hidden state for later reuse
            hidden = hidden.data

        avg_epoch_loss = np.mean(mb_losses)

        return avg_epoch_loss

    def test(self, k=20, batch_size=50):
        '''
        Args:
            k: the length of the recommendation list
            batch_size: size of the session-parallel mini-batch to use during the testing
        Returns:
            avg_recall: mean of the Recall@K over the session-parallel mini-batches
            avg_mrr: mean of the MRR@K over the session-parallel mini-batches
        '''
        # set the gru layer into inference mode
        if self.gru.training:
            self.gru.switch_mode()
        
        recalls = []
        mrrs = []
        hidden = self.gru.init_hidden().data

        for input, target, hidden in G.generate_batch(df=self.df_test,
                                                      batch_size=batch_size,
                                                      hidden=hidden,
                                                      training=False,
                                                      time_sort=self.time_sort):
            if self.use_cuda:
                input = input.cuda()
                target = target.cuda()
            # Embed the input
            embedded = self.gru.emb(input, volatile=True)
            # forward propagation
            logit, hidden = self.gru(embedded, target, hidden)
            # Evaluate the results
            recall, mrr = E.evaluate(logit, target, k)
            recalls.append(recall)
            mrrs.append(mrr)

        avg_recall = np.mean(recalls)
        avg_mrr = np.mean(mrrs)
        
        # reset the gru to a training mode
        self.gru.switch_mode()

        return avg_recall, avg_mrr

    def init_data(self, df_train, df_test, session_key, time_key, item_key):
        '''
        Initialize the training data and test data
        '''

        # Specify the identifiers
        self.session_key = session_key
        self.time_key = time_key
        self.item_key = item_key

        # Initialize the dataframes into adequate forms
        self.df_train = self.init_df(df_train, session_key, time_key, item_key)
        self.df_test = self.init_df(df_test, session_key, time_key, item_key, iids=self.df_train[item_key].unique())

    @staticmethod
    def init_df(df, session_key, time_key, item_key, iids=None):
        '''
        Initialize a dataframe.
        Involves the following steps:
            1) Adding item indices to the columns
            2) Sort the df

        Args:
            session_key: session identifier
            time_key: timestamp
            item_key: item identifier
            iids: unique item ids. Should be `None` if the df is a training set, and should include the
                  ids for the items included in the training set if the df is a test set.
        '''

        # add item index column named "iidx" to the df
        if iids is None: iids = df[item_key].unique()  # unique item ids
        item2idx = pd.Series(data=np.arange(len(iids)), index=iids)
        df = pd.merge(df,
                      pd.DataFrame({item_key: iids,
                                    'iidx': item2idx[iids].values}),
                      on=item_key,
                      how='inner')

        # Sort the df by time, and then by session ID. That is, df is sorted by session ID and
        # clicks within a session are next to each other, where the clicks within a session are time-ordered.
        df.sort_values([session_key, time_key], inplace=True)

        return df