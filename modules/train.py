import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pathlib import Path
from modules.optimizer import Optimizer
from modules.loss import LossFunction
from modules.model import GRU

class GRU4REC:
    def __init__(self, input_size, hidden_size, output_size, optimizer_type = 'Adagrad',
                 lr = .05, loss_type = 'TOP1', clip_grad = -1, dropout_input=.0,
                 dropout_hidden = .5, batch_size = 50, use_cuda = True):
        '''
        Args:
            input_size (int): dimension of the gru input variables
            hidden_size (int): dimension of the gru hidden units
            output_size (int): dimension of the gru output variables
            optimizer_type (str): optimizer type for GRU weights
            lr (float): learning rate for the optimizer
            loss_type (str): type of the loss function to use
            clip_grad (float): clip the gradient norm at clip_grad. No clipping if clip_grad = -1
            dropout_input (float): dropout probability for the input layer
            dropout_hidden (float): dropout probability for the hidden layer
            batch_size (int): mini-batch size
            use_cuda (bool): whether you want to use cuda or not
        '''
        
        # Initialize the GRU Layer
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.gru = GRU(input_size, hidden_size, output_size,
                       dropout_input = dropout_input,
                       dropout_hidden = dropout_hidden,
                       use_cuda = use_cuda)
        # Initialize the optimizer
        self.optimizer = Optimizer(self.gru.parameters(), optimizer_type)
        # Initialize the loss function
        self.loss_fn = LossFunction(loss_type, use_cuda)
        # gradient clipping(optional)
        self.clip_grad = clip_grad 
        
    def train(self, df, session_key, time_key, item_key, n_epochs=10, save_dir='./models', model_name='GRU4REC'):
        df, click_offsets, session_idx_arr = GRU4REC.init_data(df, session_key, time_key, item_key)
        # Time the training process
        start_time = time.time()
        for epoch in range(n_epochs):
            loss = self.run_epoch(df, click_offsets, session_idx_arr)
            end_time = time.time()
            wall_clock = (end_time - start_time) / 60
            print(f'Epoch:{epoch+1:2d}/Loss:{loss:0.3f}/TrainingTime:{wall_clock:0.3f}(min)')
            start_time = time.time()
            
            # Store the intermediate model
            save_dir = Path(save_dir)
            model_fname = f'{model_name}_epoch{epoch:d}'
            torch.save(self.gru.state_dict(), save_dir/model_fname)
        
    def run_epoch(self, df, click_offsets, session_idx_arr):
        mb_losses = []
        # initializations
        iters = np.arange(self.batch_size)
        maxiter = iters.max()
        start = click_offsets[session_idx_arr[iters]]
        end = click_offsets[session_idx_arr[iters]+1]
        # initialize the hidden state
        hidden = self.gru.init_hidden().data
        
        optimizer = self.optimizer
        
        # Start the training loop
        finished = False
        n = 0
        while not finished:
            minlen = (end-start).min()
            # Item indices(for embedding) for clicks where the first sessions start
            idx_target = df.iidx.values[start]
            for i in range(minlen - 1):
                # Build inputs, targets, and hidden states
                idx_input = idx_target
                idx_target = df.iidx.values[start + i + 1]
                input = torch.LongTensor(idx_input) #(B) At first, input is a Tensor
                target = Variable(torch.LongTensor(idx_target)) #(B)
                if self.use_cuda:
                    input = input.cuda()
                    target = target.cuda()
                # Now, convert into an embedded Variable
                embedded = self.gru.emb(input)
                hidden = Variable(hidden)
                
                # Go through the GRU layer
                logit, hidden = self.gru(embedded, target, hidden)
                
                # Calculate the mini-batch loss
                mb_loss = self.loss_fn(logit, target)
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
                
            # click indices where a particular session meets second-to-last element
            start = start + (minlen - 1)
            # see if how many sessions should terminate
            mask = np.arange(len(iters))[(end-start)<=1]
            for idx in mask:
                maxiter += 1
                if maxiter >= len(click_offsets)-1:
                    finished = True
                    break
                # update the next starting/ending point
                iters[idx] = maxiter
                start[idx] = click_offsets[session_idx_arr[maxiter]]
                end[idx] = click_offsets[session_idx_arr[maxiter]+1]
            
            # reset the rnn hidden state to zero after transition
            if len(mask)!= 0:
                hidden[:,mask,:] = 0

        avg_epoch_loss = np.mean(mb_losses)
        
        return avg_epoch_loss
        
    @staticmethod
    def init_data(df, session_key, time_key, item_key):
        
        '''
        Initialize the training data, carrying out several steps
        that are necessary for the training
        '''
    
        # add item indices to the dataframe
        df = GRU4REC.add_item_indices(df, item_key)

        '''
        Sort the df by time, and then by session ID.
        That is, df is sorted by session ID and 
        clicks within a session are next to each other,
        where the clicks within a session are time-ordered.
        '''
        df.sort_values([session_key, time_key], inplace = True)

        click_offsets = GRU4REC.get_click_offsets(df, session_key)
        session_idx_arr = GRU4REC.order_session_idx_by_starting_time(df, session_key, time_key)

        return df, click_offsets, session_idx_arr
        
    @staticmethod
    def add_item_indices(df, item_key):
        '''
        Adds an item index column named "iidx" to the df.

        Args:
            df: pd.DataFrame to add the item indices to

        Returns:
            df: copy of the original df with item indices
        '''
        iids = df[item_key].unique() # unique item ids
        item2idx = pd.Series(data=np.arange(len(iids)), index=iids)
        df = pd.merge(df,
                      pd.DataFrame({item_key:iids,
                                    'iidx':item2idx[iids].values}),
                      on=item_key,
                      how='inner')
        return df

    @staticmethod
    def get_click_offsets(df, session_key):
        '''
        Return the offsets of the beginning clicks of each session IDs,
        where the offset is calculated against the first click of the first session ID.
        '''

        offsets = np.zeros(df[session_key].nunique()+1, dtype=np.int32)
        # group & sort the df by session_key and get the offset values
        offsets[1:] = df.groupby(session_key).size().cumsum()

        return offsets

    @staticmethod
    def order_session_idx_by_starting_time(df, session_key, time_key):

        # starting time for each sessions, sorted by session IDs
        sessions_start_time = df.groupby(session_key)[time_key].min().values

        # order the session indices by session starting times
        session_idx_arr = np.argsort(sessions_start_time)

        return session_idx_arr
