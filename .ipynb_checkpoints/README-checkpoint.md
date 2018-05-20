# pyGRU4REC
- PyTorch Implementation of the GRU4REC model.
- Original paper: [Session-based Recommendations with Recurrent Neural Networks(ICLR 2016)](https://arxiv.org/pdf/1511.06939.pdf)
---

# Environment
- Python 3.6.3
- PyTorch 0.3.0.post4
- pandas 0.20.3
- numpy 1.13.3
---

# Usage

## Training / Test Set Specifications
- Filenames
    - Training set should be named as `train.tsv`
    - Test set should be named as `test.tsv`
- File Paths
    - `train.tsv`/`test.tsv` should be located under the `./data` directory. i.e. `data/train.tsv`, `data/test.tsv`
- Contents
    - `train.tsv`, `test.tsv` should be the tsv files that stores the pandas dataframes that satisfy the following requirements(without headers):
        - The 1st column of the tsv file should be the integer Session IDs
        - The 2nd column of the tsv file should be the integer Item IDs
        - The 3rd column of the tsv file should be the Timestamps

## Training/Testing using Jupyter Notebook
See `example.ipynb` for the full jupyter notebook script that
1. Loads the data
2. Trains a GRU4REC model
3. Load the trained GRU4REC model
4. Tests a GRU4REC model

## Training using `run_train.py`
- Before using `run_train.py`, I highly recommend that you to take a look at `example.ipynb` to see how the implementation works.
- Default parameters are the same as the TOP1 loss case in the [GRU4REC paper](https://arxiv.org/pdf/1511.06939.pdf).
- Intermediate models created from each training epochs will be stored to `models/`, unless specified.
- The log file will be written to `./logs/train.out`.

```
$ python run_train.py > ./logs/train.out

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
    --n_samples: The number of samples to use for the training. If -1, all samples in the training set are used.(Default: -1)
```

## Testing using `run_test.py`
```
$ python run_test.py model_file > ./logs/test.out

Args:
    model_file: name of the intermediate model under the `./models` directory. e.g. `python run_test.py GRU4REC_TOP1_Adagrad_0.01_epoch10 > ./logs/test.out`
    --loss_type: Loss function type. Should be one of 'TOP1', 'BPR', 'CrossEntropy'.(Default: 'TOP1')
    --hidden_size: The dimension of the hidden layer of the GRU.(Default: 100)
    --num_layers: The number of layers for the GRU.(Default: 1)
    --batch_size: Training batch size.(Default: 50)
    --dropout_input: Dropout probability of the input layer of the GRU.(Default: 0)
    --dropout_hidden: Dropout probability of the hidden layer of the GRU.(Default: .5)
    --optimizer_type: Optimizer type. Should be one of 'Adagrad', 'RMSProp', 'Adadelta', 'Adam', 'SGD'(Default: 'Adagrad')
    --lr: Learning rate for the optimizer.(Default: 0.01)
    --weight_decay: Weight decay for the optimizer.(Default: 0)
    --momentum: Momentum for the optimizer.(Default: 0)
    --eps: eps parameter for the optimizer.(Default: 1e-6)
    --n_epochs: The number of training epochs to run.(Default: 10)
    --time_sort: Whether to sort the sessions in the training set in a time order.(Default: False)
    --n_samples: The number of samples to use for the training. If -1, all samples in the training set are used.(Default: -1)
    ```