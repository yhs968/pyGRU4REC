import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from pathlib import Path

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers = 1,
                 dropout_hidden = .5, dropout_input = 0, batch_size = 50, use_cuda = True):
        
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_input = dropout_input
        self.dropout_hidden = dropout_hidden
        
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        
        self.onehot_buffer = self.init_emb()
        self.h2o = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout = dropout_hidden)
        
        if self.use_cuda:
            self = self.cuda()
        
    def forward(self, embedded, target, hidden):
        '''
        Args:
            embedded (B,C): embedded item indices from a session-parallel mini-batch.
            target (B,): torch.LongTensor of next item indices from a session-parallel mini-batch.
            
        Returns:
            logit (B,B): Variable that stores the logits for the items in the session-parallel mini-batch
        '''
        # Apply dropout to inputs
        p_drop = torch.Tensor(embedded.size(0),1).fill_(1 - self.dropout_input) #(B,1)
        mask = Variable(torch.bernoulli(p_drop).expand_as(embedded)) #(B,C)
        if self.use_cuda: mask = mask.cuda()
        embedded = embedded * mask #(B,C)
        embedded = embedded.unsqueeze(0) #(1,B,C)
        
        # Go through the GRU layer
        output, hidden = self.gru(embedded, hidden) #(num_layers,B,H)
        
        '''
        Sampling on the activation.
        Scores will be calculated on only the items appearing in this mini-batch.
        '''
        # self.h2o(output): (1,B,H)
        output = output.view(-1, output.size(-1)) #(B,H)
        logit = self.tanh(self.h2o(output)) #(B,C)
        logit = logit[:,target.view(-1)] #(B,B)
                    
        return logit, hidden
    
    def emb(self, input):
        '''
        Returns a one-hot vector corresponding to the input
        
        Args:
            input (B,): torch.LongTensor of item indices
            
        Returns:
            v (B,C): torch.FloatTensor of one-hot vectors
        '''
        # flush the buffer
        self.onehot_buffer.zero_()
        # fill the buffer with 1 where needed
        index = input.view(-1,1)
        self.onehot_buffer.scatter_(1, index, 1)
        
        v = Variable(self.onehot_buffer)
        
        return v.cuda() if self.use_cuda else v
    
    def init_emb(self):
        '''
        Initialize the one_hot embedding buffer for repeated one-hot embedding
        '''
        onehot_buffer = torch.FloatTensor(self.batch_size, self.output_size)
        if self.use_cuda: onehot_buffer = onehot_buffer.cuda()
        
        return onehot_buffer
    
    def init_hidden(self):
        h0 = Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
        
        return h0.cuda() if self.use_cuda else h0
