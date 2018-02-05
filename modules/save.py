# from pathlib import Path

# def save(model, PATH_TO_MODELS: str,  epoch: int):
#     '''
#     Args:
#         PATH_TO_MODELS: directory to store the models and the results
#         model: PyTorch model to save the parameters from
#         model_name (str): name of the model
#         epoch: index of the epoch
#     '''


#     if not PATH_TO_MODELS.exists():
#         PATH_TO_MODELS.mkdir()

#     subdirs = list(PATH_TO_MODELS.iterdir())
#     next_dir = str(int(max(subdirs).stem)+1) if len(subdirs)>0 else str(0)
#     next_dir = PATH_TO_MODELS/next_dir

#     PATH_ARGS = next_dir/'arguments.txt'
#     with open(PATH_ARGS, 'w') as f:
#         s = f'input_size:{input_size}\n'
#         s += f'output_size:{output_size}\n'
#         s += f'hidden_size:{hidden_size}\n'
#         s += f'num_layers:{num_layers}\n'
#         s += f'use_cuda:{use_cuda}\n'
#         s += f'time_sort:{time_sort}\n'
#         s += f'dropout_hidden:{dropout_hidden}\n'
#         s += f'dropout_input:{dropout_input}\n'
#         s += f'batch_size:{batch_size}'
#         f.write(s)

#     PATH_MODEL = next_dir/f'GRU4REC_epoch{epoch:d}'