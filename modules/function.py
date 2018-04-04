from torch.autograd import Function

class emb(Function):
    @staticmethod
    def forward(ctx, input, buffer):
        """
        Returns a one-hot vector corresponding to the input

        Args:
            input (B,): torch.LongTensor of item indices
            buffer (B,output_size): buffer that stores the one-hot vector
        Returns:
            one_hot (B,C): torch.FloatTensor of one-hot vectors
        """
        buffer.zero_()
        index = input.view(-1,1)
        one_hot = buffer.scatter_(1, index, 1)
        
        return one_hot
    
    @staticmethod
    def backward(ctx, grad_output):
        return None, None