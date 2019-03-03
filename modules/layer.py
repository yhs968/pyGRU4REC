import torch
import torch.nn as nn


class GRU(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=1,
                 p_dropout_hidden=0, p_dropout_input=0, batch_size=50, use_cuda=True):
        '''
        The GRU layer used for the whole GRU4REC model.

        Args:
            input_size (int): input layer dimension
            hidden_size (int): hidden layer dimension
            output_size (int): output layer dimension. Equivalent to the number of classes
            num_layers (int): the number of GRU layers
            p_dropout_hidden (float): dropout probability for the GRU hidden layers
            p_dropout_input (float): dropout probability for the GRU input layer
            batch_size (int): size of the training batch.(required for producing one-hot encodings efficiently)
            use_cuda (bool): whether to use cuda or not
            training (bool): whether to set the GRU module to training mode or not. If false, parameters will not be updated.
        '''

        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.p_dropout_input = p_dropout_input
        self.dropout_hidden = nn.Dropout(p_dropout_hidden)

        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.onehot_buffer = self.init_emb()  # the buffer where the one-hot encodings will be produced from
        self.h2o = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=p_dropout_hidden if num_layers > 1 else 0)
        
        self = self.to(self.device)
        

    def forward(self, input, target, hidden):
        '''
        Args:
            input (B,): a batch of item indices from a session-parallel mini-batch.
            target (B,): torch.LongTensor of next item indices from a session-parallel mini-batch.
            
        Returns:
            logit (B,C): Variable that stores the logits for the next items in the session-parallel mini-batch
            hidden: GRU hidden state
        '''
        embedded = self.onehot_encode(input)
        if self.training and self.p_dropout_input > 0: embedded = self.input_dropout(embedded)
        embedded = embedded.unsqueeze(0)  # (1,B,C)

        # Go through the GRU layer
        output, hidden = self.gru(embedded, hidden)  # (num_layers,B,H)
        output = output.view(-1, output.size(-1))  # (B,H)
        output = self.dropout_hidden(output) # hidden layer dropout
        logit = self.tanh(self.h2o(output))  # (B,C)

        return logit, hidden
    
    
    def input_dropout(self, input):
        p_drop = torch.Tensor(input.size(0), 1).fill_(1 - self.p_dropout_input)  # (B,1)
        mask = torch.bernoulli(p_drop).expand_as(input)/(1-self.p_dropout_input) # (B,C)
        mask = mask.to(self.device)
        input = input * mask  # (B,C)
        
        return input
        

    def init_emb(self):
        '''
        Initialize the one_hot embedding buffer, which will be used for producing the one-hot embeddings efficiently
        '''
        onehot_buffer = torch.FloatTensor(self.batch_size, self.output_size)
        onehot_buffer = onehot_buffer.to(self.device)

        return onehot_buffer
    
    
    def onehot_encode(self, input):
        """
        Returns a one-hot vector corresponding to the input

        Args:
            input (B,): torch.LongTensor of item indices
            buffer (B,output_size): buffer that stores the one-hot vector
        Returns:
            one_hot (B,C): torch.FloatTensor of one-hot vectors
        """
        
        self.onehot_buffer.zero_()
        index = input.view(-1,1)
        one_hot = self.onehot_buffer.scatter_(1, index, 1)
        
        return one_hot

    
    def init_hidden(self):
        '''
        Initialize the hidden state of the GRU
        '''
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)

        return h0