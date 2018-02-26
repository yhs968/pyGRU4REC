import time
import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
from pathlib import Path
from modules.optimizer import Optimizer
from modules.loss import LossFunction
from modules.layer import GRU
from modules.evaluate import evaluate

class GRU4REC:
    def __init__(self, input_size, hidden_size, output_size, num_layers=1,
                 optimizer_type='Adagrad', lr=.05, weight_decay=0,
                 momentum=0, eps=1e-6, loss_type='TOP1',
                 clip_grad=-1, dropout_input=.0, dropout_hidden=.5,
                 batch_size=50, use_cuda=True, time_sort=False, pretrained=None):

        """ The GRU4REC model

        Args:
            input_size (int): dimension of the gru input variables
            hidden_size (int): dimension of the gru hidden units
            output_size (int): dimension of the gru output variables
            num_layers (int): the number of layers in the GRU
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
        """

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

    def train(self, df, session_key, time_key, item_key, n_epochs=10, save_dir='./models', model_name='GRU4REC'):
        """
        Train the GRU4REC model on a pandas dataframe for several training epochs,
        and store the intermediate models to the user-specified directory.

        Args:
            df (pd.DataFrame): training dataset
            session_key (str): session ID
            time_key (str): time ID
            item_key (str): item ID
            n_epochs (int): the number of training epochs to run
            save_dir (str): the path to save the intermediate trained models
            model_name (str): name of the model
        """
        df, click_offsets, session_idx_arr = GRU4REC.init_data(df, session_key, time_key, item_key,
                                                               time_sort=self.time_sort)
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
            if not save_dir.exists(): save_dir.mkdir()

            model_fname = f'{model_name}_{self.loss_type}_{self.optimizer_type}_{self.lr}_epoch{epoch+1:d}'
            torch.save(self.gru.state_dict(), save_dir/model_fname)

    def run_epoch(self, df, click_offsets, session_idx_arr):
        """ Runs a single training epoch """
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
        while not finished:
            minlen = (end-start).min()
            # Item indices(for embedding) for clicks where the first sessions start
            idx_target = df.item_idx.values[start]
            for i in range(minlen - 1):
                # Build inputs, targets, and hidden states
                idx_input = idx_target
                idx_target = df.item_idx.values[start + i + 1]
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

            # click indices where a particular session meets second-to-last element
            start = start + (minlen - 1)
            # figure out how many sessions should terminate
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
            if len(mask) != 0:
                hidden[:, mask, :] = 0

        avg_epoch_loss = np.mean(mb_losses)

        return avg_epoch_loss

    def predict(self, input, target, hidden):
        """ Forward propagation for testing

        Args:
            input (B,C): torch.LongTensor. The one-hot embedding for the item indices
            target (B,): a Variable that stores the indices for the next items
            hidden: previous hidden state

        Returns:
            logits (B,C): logits for the next items
            hidden: next hidden state
        """
        # convert the item indices into embeddings
        embedded = self.gru.emb(input, volatile=True)
        hidden = Variable(hidden, volatile=True)
        # forward propagation
        logits, hidden = self.gru(embedded, target, hidden)

        return logits, hidden

    def test(self, df_train, df_test, session_key, time_key, item_key,
             k=20, batch_size=50):
        """ Model evaluation

        Args:
            df_train (pd.DataFrame): training set required to retrieve the training item indices.
            df_test (pd.DataFrame): test set
            session_key (str): session ID
            time_key (str): time ID
            item_key (str): item ID
            k (int): the length of the recommendation list
            batch_size (int): testing batch_size

        Returns:
            avg_recall: mean of the Recall@K over the session-parallel mini-batches
            avg_mrr: mean of the MRR@K over the session-parallel mini-batches
        """
        # set the gru layer into inference mode
        if self.gru.training:
            self.gru.switch_mode()

        recalls = []
        mrrs = []

        # initializations
        # Build item2idx from train data.
        iids = df_train[item_key].unique() # unique item ids
        item2idx = pd.Series(data=np.arange(len(iids)), index=iids)
        df_test = pd.merge(df_test,
                           pd.DataFrame({item_key: iids,
                                         'item_idx': item2idx[iids].values}),
                           on=item_key,
                           how='inner')
        # Sort the df by time, and then by session ID.
        df_test.sort_values([session_key, time_key], inplace=True)
        # Return the offsets of the beginning clicks of each session IDs
        click_offsets = GRU4REC.get_click_offsets(df_test, session_key)
        session_idx_arr = GRU4REC.order_session_idx(df_test, session_key, time_key, time_sort=self.time_sort)

        iters = np.arange(batch_size)
        maxiter = iters.max()
        start = click_offsets[session_idx_arr[iters]]
        end = click_offsets[session_idx_arr[iters]+1]
        hidden = self.gru.init_hidden().data

        # Start the training loop
        finished = False
        while not finished:
            minlen = (end-start).min()
            # Item indices(for embedding) for clicks where the first sessions start
            idx_target = df_test.item_idx.values[start]
            for i in range(minlen - 1):
                # Build inputs, targets, and hidden states
                idx_input = idx_target
                idx_target = df_test.item_idx.values[start + i + 1]
                input = torch.LongTensor(idx_input) #(B) At first, input is a Tensor
                target = Variable(torch.LongTensor(idx_target), volatile=True)  # (B)
                if self.use_cuda:
                    input = input.cuda()
                    target = target.cuda()

                logit, hidden = self.predict(input, target, hidden)
                recall, mrr = evaluate(logit, target, k)
                recalls.append(recall)
                mrrs.append(mrr)

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

        avg_recall = np.mean(recalls)
        avg_mrr = np.mean(mrrs)

        # reset the gru to a training mode
        self.gru.switch_mode()

        return avg_recall, avg_mrr

    @staticmethod
    def init_data(df, session_key, time_key, item_key, time_sort):
        """
        Initialize the data.
        """

        # add item indices to the dataframe
        df = GRU4REC.add_item_indices(df, item_key)

        """
        Sort the df by time, and then by session ID. That is, df is sorted by session ID and 
        clicks within a session are next to each other, where the clicks within a session are time-ordered.
        """
        df.sort_values([session_key, time_key], inplace=True)

        click_offsets = GRU4REC.get_click_offsets(df, session_key)
        session_idx_arr = GRU4REC.order_session_idx(df, session_key, time_key, time_sort=time_sort)

        return df, click_offsets, session_idx_arr

    @staticmethod
    def add_item_indices(df, item_key):
        """
        Adds an item index column named "item_idx" to the df.

        Args:
            df: pd.DataFrame to add the item indices to

        Returns:
            df: copy of the original df with item indices
        """
        iids = df[item_key].unique() # unique item ids
        item2idx = pd.Series(data=np.arange(len(iids)), index=iids)
        df = pd.merge(df,
                      pd.DataFrame({item_key: iids,
                                    'item_idx': item2idx[iids].values}),
                      on=item_key,
                      how='inner')
        return df

    @staticmethod
    def get_click_offsets(df, session_key):
        """
        Return the offsets of the beginning clicks of each session IDs,
        where the offset is calculated against the first click of the first session ID.
        """

        offsets = np.zeros(df[session_key].nunique()+1, dtype=np.int32)
        # group & sort the df by session_key and get the offset values
        offsets[1:] = df.groupby(session_key).size().cumsum()

        return offsets

    @staticmethod
    def order_session_idx(df, session_key, time_key, time_sort=False):
        """ Order the session indices """

        if time_sort:
            # starting time for each sessions, sorted by session IDs
            sessions_start_time = df.groupby(session_key)[time_key].min().values
            # order the session indices by session starting times
            session_idx_arr = np.argsort(sessions_start_time)
        else:
            session_idx_arr = np.arange(df[session_key].nunique())

        return session_idx_arr
