import torch.nn as nn
from collections import OrderedDict
from loldraftassist.ml import device, batch_gd, evaluate, train_and_evaluate, cross_val, shuffle_and_split
from loldraftassist.champ_data import num_champs
from loldraftassist.soloq.data import db_to_tensor


patch = '11_3soloq'

# To train the model with champion input vectors and win confidence outputs,
# used the following parameters:
# optimizer=optim.Adam      lrate=1.5e-3        k=100       loss_fn=nn.CrossEntropyLoss()
# iters=2500        network arch=model_arch
model_arch = OrderedDict([('lin1', nn.Linear(2 * num_champs, 232)),
                        ('relu1', nn.ReLU()),
                        ('lin2', nn.Linear(232, 2)),
                        ('batchnorm', nn.BatchNorm1d(2))])

if __name__ == '__main__':
    fullX, fullY = db_to_tensor(f'db/{patch}.db')
    train_and_evaluate(fullX, fullY, model_arch, f'models/{patch}')
