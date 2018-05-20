import numpy as np
import torch
from torch.autograd import Variable


def generate_batch(df, session_key, time_key, batch_size, hidden, training=True, time_sort=False):
    '''
    Args:
         df (pd.DataFrame): dataframe to generate the batches from
         batch_size (int): size of the batch
         hidden (torch.FloatTensor): initial hidden state
         training (bool): whether to generate the batches in the training mode. If False, Variables will be created with the flag `volatile = True`.
         time_sort (bool): whether to sort the sessions by time when generating batches
    Returns:
        input (torch.LongTensor)
        target (torch.autograd.Variable)
        hidden (torch.autograd.Variable)

    '''
    # initializations
    click_offsets = get_click_offsets(df, session_key=session_key)
    session_idx_arr = order_session_idx(df, session_key=session_key, time_key=time_key, time_sort=time_sort)

    iters = np.arange(batch_size)
    maxiter = iters.max()
    start = click_offsets[session_idx_arr[iters]]
    end = click_offsets[session_idx_arr[iters] + 1]
    finished = False
    hidden = hidden

    while not finished:
        minlen = (end - start).min()
        # Item indices(for embedding) for clicks where the first sessions start
        idx_target = df.iidx.values[start]
        # Train until any session ends
        for i in range(minlen - 1):
            # Build inputs, targets, and hidden states
            idx_input = idx_target
            idx_target = df.iidx.values[start + i + 1]
            input = torch.LongTensor(idx_input) #(B) At first, input is a Tensor
            if training:
                target = Variable(torch.LongTensor(idx_target))  # (B)
                hidden = Variable(hidden)
            else:
                target = Variable(torch.LongTensor(idx_target), volatile=True)  # (B)
                hidden = Variable(hidden, volatile=True)

            yield input, target, hidden

            # Detach the hidden state for later reuse
            hidden = hidden.data

        # Tasks to carry out after a particular session terminates

        ## click indices where a particular session meets second-to-last element
        start = start + (minlen - 1)
        ## figure out how many sessions should terminate
        mask = np.arange(len(iters))[(end - start) <= 1]
        for idx in mask:
            maxiter += 1
            if maxiter >= len(click_offsets) - 1:
                finished = True
                break
            # update the next starting/ending point
            iters[idx] = maxiter
            start[idx] = click_offsets[session_idx_arr[maxiter]]
            end[idx] = click_offsets[session_idx_arr[maxiter] + 1]

        ## reset the rnn hidden state to zero after transition
        if len(mask) != 0:
            hidden[:, mask, :] = 0


def get_click_offsets(df, session_key):
    '''
    Return the offsets of the beginning clicks of each session IDs,
    where the offset is calculated against the first click of the first session ID.
    '''

    offsets = np.zeros(df[session_key].nunique()+1, dtype=np.int32)
    # group & sort the df by session_key and get the offset values
    offsets[1:] = df.groupby(session_key).size().cumsum()

    return offsets


def order_session_idx(df, session_key, time_key, time_sort=False):
    '''
    Order the session indices
    '''

    if time_sort:
        # starting time for each sessions, sorted by session IDs
        sessions_start_time = df.groupby(session_key)[time_key].min().values
        # order the session indices by session starting times
        session_idx_arr = np.argsort(sessions_start_time)
    else:
        session_idx_arr = np.arange(df[session_key].nunique())

    return session_idx_arr