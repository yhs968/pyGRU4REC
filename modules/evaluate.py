import torch

def get_recall(indices, targets):
    '''
    Args:
        indices:(Bxk) torch.LongTensor
        targets:(B) torch.LongTensor
    '''
    targets = targets.view(-1,1).expand_as(indices) # (Bxk)
    hits = (targets == indices).nonzero()
    if len(hits) == 0: return 0
    n_hits = (targets == indices).nonzero()[:,:-1].size(0)
    
    return n_hits / targets.size(0)

def get_mrr(indices, targets):
    '''
    Args:
        indices:(Bxk) torch.LongTensor
        targets:(B) torch.LongTensor
    '''
    targets = targets.view(-1,1).expand_as(indices)
    # ranks of the targets, if it appears in your indices
    hits = (targets == indices).nonzero()
    if len(hits) == 0: return 0
    ranks = hits[:,-1] + 1
    ranks = ranks.float()
    rranks = torch.reciprocal(ranks)
    
    return torch.sum(rranks).data / targets.size(0)

def evaluate(logits, targets, k = 20):
    '''
    Args:
        logits (B,k): torch.FloatTensor
        targets (B): torch.LongTensor
    '''
    _, indices = torch.topk(logits, k, -1)
    recall = get_recall(indices, targets)
    mrr = get_mrr(indices, targets)
    
    return recall, mrr