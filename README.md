# ※ Disclaimer
- A critical bug is suspected in this code. See the update log for 03/02/2019.
- I recommend to refer to the [original author's implementation](https://github.com/hidasib/GRU4Rec).

# pyGRU4REC
- PyTorch Implementation of the GRU4REC model.
- Original paper: [Session-based Recommendations with Recurrent Neural Networks(ICLR 2016)](https://arxiv.org/pdf/1511.06939.pdf)
- This code is mostly a PyTorch re-implementation of the [original Theano code written by the authors of the GRU4REC paper](https://github.com/hidasib/GRU4Rec). What I did are as below:
    - Replaced the Theano components with PyTorch
    - Simplifying and Cleaning the session-parallel mini-batch generation code

# Update(03/02/2019)
- PyTorch 1.0 migration
- Code optimization
    - replaced the pandas dataframe component with numpy array for speedup
- **Somehow, the model failed to retrieve the original performance reported in the first version of the implementation.(potential bug)**
    - However, I do not have plans to fix this bug in the near future due to contraint in time & computational resource.

# Update(05/20/2018)
- PyTorch 0.4.0 support
    - The code is now compatible with PyTorch >= 0.4.0
- Code cleanup
    - Removed redundant pieces of code thanks to simpler API of PyTorch 0.4.0
    - Improved the readablility of the confusing rnn updating routine
    - Improved the readability of the training/testing routine
- Optimization
    - Testing code is now much faster than before

# Environment
- **PyTorch 0.4.0**
- Python 3.6.4
- pandas 0.22.0
- numpy 1.14.0
---

# Usage

## Training / Test Set Specifications
- Filenames
    - Training set should be named as `train.tsv`
    - Test set should be named as `test.tsv`
- File Paths
    - `train.tsv`, `test.tsv` should be located under the `data` directory. i.e. `data/train.tsv`, `data/test.tsv`
- Contents
    - `train.tsv`, `test.tsv` should be the tsv files that stores the pandas dataframes that satisfy the following requirements(without headers):
        - The 1st column of the tsv file should be the integer Session IDs
        - The 2nd column of the tsv file should be the integer Item IDs
        - The 3rd column of the tsv file should be the Timestamps

## Training/Testing using Jupyter Notebook
See `example.ipynb` for the full jupyter notebook script that
1. Loads the data
2. Trains & tests a GRU4REC model
3. Loads & tests a pretrained GRU4REC model

## Training & Testing using `run_train.py`
- Before using `run_train.py`, I highly recommend that you to take a look at `example.ipynb` to see how the implementation works in general.
- Default parameters are the same as the TOP1 loss case in the [GRU4REC paper](https://arxiv.org/pdf/1511.06939.pdf).
- Intermediate models created from each training epochs will be stored to `models/`, unless specified.
- The log file will be written to `logs/train.out`.

```
$ python run_train.py > logs/train.out

Args:
    --loss_type: Loss function type. Should be one of the 'TOP1', 'BPR', 'CrossEntropy'.(Default: 'TOP1')
    --model_name: The prefix for the intermediate models that will be stored during the training.(Default: 'GRU4REC')
    --hidden_size: The dimension of the hidden layer of the GRU.(Default: 100)
    --num_layers: The number of layers for the GRU.(Default: 1)
    --batch_size: Training batch size.(Default: 50)
    --dropout_input: Dropout probability of the input layer of the GRU.(Default: 0)
    --dropout_hidden: Dropout probability of the hidden layer of the GRU.(Default: .5)
    --optimizer_type: Optimizer type. Should be one of the 'Adagrad', 'RMSProp', 'Adadelta', 'Adam', 'SGD'(Default: 'Adagrad')
    --lr: Learning rate for the optimizer.(Default: 0.01)
    --weight_decay: Weight decay for the optimizer.(Default: 0)
    --momentum: Momentum for the optimizer.(Default: 0)
    --eps: eps parameter for the optimizer.(Default: 1e-6)
    --n_epochs: The number of training epochs to run.(Default: 10)
    --time_sort: Whether to sort the sessions in the dataset in a time order.(Default: False)
```

## Reproducing the results of the original paper
<S>- The results from this PyTorch Implementation gives a slightly better result compared to the [original code](https://github.com/hidasib/GRU4Rec) that was written in Theano. I guess this comes from the difference between Theano and PyTorch & the fact that dropout has no effect in my single-layered PyTorch GRU implementation.
- **The results were reproducible within only 2 or 3 epochs**, unlike the [original Theano implementation](https://github.com/hidasib/GRU4Rec/blob/master/gru4rec.py) which runs for 10 epochs by default.</S>
```
$ bash run_train.sh
```
- (03/02/2019) The results are now far worse than the original GRU4REC paper(recall 0.33 mrr 0.183). I suspect that there is a critical bug in the updated version of the code. However, due to the constraint in time & computational resource, I'm not planning to fix this in near future.
