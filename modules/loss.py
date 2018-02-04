import torch
import torch.nn as nn
from torch.autograd import Variable

class LossFunction(nn.Module):
    def __init__(self, loss_type='TOP1', use_cuda = True):
        super().__init__()
        self.loss_type = loss_type
        self.use_cuda = use_cuda
        if loss_type == 'CrossEntropy':
            self._loss_fn = nn.CrossEntropyLoss()
        elif loss_type == 'TOP1':
            self._loss_fn = TOP1Loss()
        elif loss_type == 'BPR':
            self._loss_fn = BPRLoss()
        else: raise NotImplementedError
        
    def forward(self, *args):
        if self.loss_type in {'TOP1','BPR'}:
            logit = args[0]
            return self._loss_fn(logit)
        elif self.loss_type == 'CrossEntropy':
            # cross-entropy loss with n_classes = batch_size
            logit = args[0]
            batch_size = logit.size(1)
            
            target = Variable(torch.arange(batch_size).type(torch.LongTensor))
            if self.use_cuda: target = target.cuda()
                
            return self._loss_fn(logit, target)
        else:
            raise NotImplementedError
    
class BPRLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.logsigmoid = nn.LogSigmoid()
    
    def forward(self, logit):
        '''
        See Balazs Hihasi(ICLR 2016), pg.5
        
        Args:
            logit (BxB): Variable that stores the logits for the items in the mini-batch
                         The first dimension corresponds to the batches, and the second
                         dimension corresponds to sampled number of items to evaluate
        '''
        
        # differences between the item scores
        '''
        logit.diag().diag() builds a diagonal matrix with the diagonal elements
        in the logit
        '''
        difference = logit.diag().diag()-logit
        # final loss
        loss = -torch.mean(self.logsigmoid(difference))
        
        return loss
    
class TOP1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, logit):
        '''
        See Balazs Hihasi(ICLR 2016), pg.5
        
        Args:
            logit (BxB): Variable that stores the logits for the items in the mini-batch
                         The first dimension corresponds to the batches, and the second
                         dimension corresponds to sampled number of items to evaluate
        '''
        # differences between the item scores
        '''
        logit.diag().diag() builds a diagonal matrix with the diagonal elements
        in the logit
        '''
        difference = -(logit.diag().diag()-logit)
        # final loss
        loss = torch.mean(self.sigmoid(difference)) + torch.mean(self.sigmoid(logit**2))
        
        return loss