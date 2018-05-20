import time
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from modules.optimizer import Optimizer
from modules.loss import LossFunction
from modules.layer import GRU
import modules.evaluate as E
from modules.data import SessionDataset, SessionDataLoader


class GRU4REC:
    def __init__(self, input_size, hidden_size, output_size, num_layers=1,
                 optimizer_type='Adagrad', lr=.01, weight_decay=0,
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
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        if pretrained is None:
            self.gru = GRU(input_size, hidden_size, output_size, num_layers,
                           dropout_input=dropout_input,
                           dropout_hidden=dropout_hidden,
                           batch_size=batch_size,
                           use_cuda=use_cuda)
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
        
        
    def run_epoch(self, dataset, k=20, training=True):
        """ Run a single training epoch """
        start_time = time.time()
        
        # initialize
        losses = []
        recalls = []
        mrrs = []
        optimizer = self.optimizer
        hidden = self.gru.init_hidden()
        if not training:
            self.gru.eval()
        device = self.device
        
        def reset_hidden(hidden, mask):
            """Helper function that resets hidden state when some sessions terminate"""
            if len(mask) != 0:
                hidden[:, mask, :] = 0
            
            return hidden

        # Start the training loop
        loader = SessionDataLoader(dataset, batch_size=self.batch_size)

        for input, target, mask in loader:
            input = input.to(device)
            target = target.to(device)
            # reset the hidden states if some sessions have just terminated
            hidden = reset_hidden(hidden, mask).detach()
            # Go through the GRU layer
            logit, hidden = self.gru(input, target, hidden)
            # Output sampling
            logit_sampled = logit[:, target.view(-1)]
            # Calculate the mini-batch loss
            loss = self.loss_fn(logit_sampled)
            with torch.no_grad():
                recall, mrr = E.evaluate(logit, target, k)
            losses.append(loss.item())         
            recalls.append(recall)
            mrrs.append(mrr)
            # Gradient Clipping(Optional)
            if self.clip_grad != -1:
                for p in self.gru.parameters():
                    p.grad.data.clamp_(max=self.clip_grad)
            # Mini-batch GD
            if training:
                # Backprop
                loss.backward()
                optimizer.step()
                optimizer.zero_grad() # flush the gradient after the optimization

        results = dict()
        results['loss'] = np.mean(losses)
        results['recall'] = np.mean(recalls)
        results['mrr'] = np.mean(mrrs)
        
        end_time = time.time()
        results['time'] = (end_time - start_time) / 60
        
        if not training:
            self.gru.train()

        return results
    
    
    def train(self, dataset, k=20, n_epochs=10, save_dir='./models', save=True, model_name='GRU4REC'):
        """
        Train the GRU4REC model on a pandas dataframe for several training epochs,
        and store the intermediate models to the user-specified directory.

        Args:
            n_epochs (int): the number of training epochs to run
            save_dir (str): the path to save the intermediate trained models
            model_name (str): name of the model
        """
        print(f'Training {model_name}...')
        for epoch in range(n_epochs):
            results = self.run_epoch(dataset, k=k, training=True)
            results = [f'{k}:{v:.3f}' for k, v in results.items()]
            print(f'epoch:{epoch+1:2d}/{"/".join(results)}')
            
            # Store the intermediate model
            if save:
                save_dir = Path(save_dir)
                if not save_dir.exists(): save_dir.mkdir()
                model_fname = f'{model_name}_{self.loss_type}_{self.optimizer_type}_{self.lr}_epoch{epoch+1:d}'
                torch.save(self.gru.state_dict(), save_dir/model_fname)
    

    def test(self, dataset, k=20):
        """ Model evaluation

        Args:
            k (int): the length of the recommendation list

        Returns:
            avg_loss: mean of the losses over the session-parallel minibatches
            avg_recall: mean of the Recall@K over the session-parallel mini-batches
            avg_mrr: mean of the MRR@K over the session-parallel mini-batches
            wall_clock: time took for testing
        """
        results = self.run_epoch(dataset, k=k, training=False)
        results = [f'{k}:{v:.3f}' for k, v in results.items()]
        print(f'Test result: {"/".join(results)}')
    

