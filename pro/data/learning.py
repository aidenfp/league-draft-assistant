import torch as t
import torch.nn as nn
from collections import OrderedDict
from main.ml import device, batch_gd, evaluate, train_and_evaluate, cross_val, shuffle_and_split
from pro.data.data import num_champs, db_to_tensor

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
print('Device: ', device)
if device == t.device("cuda:0"):
    t.cuda.device(device)

model_arch = OrderedDict([('lin1', nn.Linear(10 * num_champs, 15 * num_champs)),
                          ('relu1', nn.ReLU()),
                          ('lin2', nn.Linear(15 * num_champs, 5 * num_champs)),
                          ('relu2', nn.ReLU()),
                          ('lin3', nn.Linear(5 * num_champs, num_champs//2)),
                          ('relu3', nn.ReLU()),
                          ('lin4', nn.Linear(num_champs//2, 2)),
                          ('batchnorm', nn.BatchNorm1d(2))])


if __name__ == '__main__':
    fullX, fullY = db_to_tensor('db/2021pro.db', ('11.02', '11.03'))
    train_and_evaluate(fullX, fullY, model_arch, '../models/pro2021_0213')
