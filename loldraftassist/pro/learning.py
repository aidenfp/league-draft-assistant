import torch as t
import torch.nn as nn
import torch.optim as optim
import sqlite3 as sql
import pickle
from collections import OrderedDict
from loldraftassist.ml import device, batch_gd, evaluate, train_and_evaluate, cross_val, shuffle_and_split
from loldraftassist.pro.data import db_to_tensor
from loldraftassist.champ_data import num_champs

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
print('Device: ', device)
if device == t.device("cuda:0"):
    t.cuda.device(device)

db = 'loldraftassist/pro/db/arch_tests.db'


# returns OrderedDict of neural network architecture
def make_arch(num_layers, initial_factor, scaling, non_linear_func, dropout=None):
    out = []
    prev_size = 10 * num_champs
    for i in range(num_layers):
        if i == 0:
            new_size = int(initial_factor * prev_size)
        elif i == num_layers-1:
            new_size = 2
        else:
            new_size = int(prev_size * scaling)
        out.append((f'lin{i}', nn.Linear(prev_size, new_size)))
        out.append((f'nonlin{i}', non_linear_func()))
        if dropout:
            out.append((f'drop{i}', nn.Dropout(p=dropout)))
        prev_size = new_size
    out.append(('batchnorm', nn.BatchNorm1d(2)))
    return OrderedDict(out)


# performs a cross-validation test on architecture with given parameters and records its score and optionally saves
# the results to a sql database
def test_arch(num_layers, initial_factor, scaling, non_linear_func, dropout, optim, lrate, batch_size, x, y, save=True):
    global db
    score = cross_val(make_arch(num_layers, initial_factor, scaling, non_linear_func, dropout), optim, x, y, lam=lrate, b_size=batch_size)
    if save:
        conn = sql.connect(db)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS tests_table (num_layers int, initial_factor int, scaling float, non_linear_func text, dropout float, optim text, lrate float, batch_size float, score float);''')
        c.execute('''INSERT into tests_table VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', (num_layers, initial_factor, scaling, str(non_linear_func), dropout, str(optim), lrate, batch_size, score))
        conn.commit()
        conn.close()
    return score


def test_archs(sizes, initials, scalings, nl_funcs, dropouts, optims, lrates, batch_sizes, x, y):
    try:
        for size in sizes:
            for initial in initials:
                for scaling in scalings:
                    for nl_func in nl_funcs:
                        for dropout in dropouts:
                            for optim in optims:
                                for lrate in lrates:
                                    for batch_size in batch_sizes:
                                        print(f"Beginning test on network with the following parameters\nSize: {size}\nInitial scaling: {initial}\nActivation: {nl_func}\n Dropout Probability: {dropout}\n Optimizer: {optim}\n Learning rate: {lrate}\nBatch size: {batch_size}")
                                        test_arch(size, initial, scaling, nl_func, dropout, optim, lrate, batch_size, x, y)
    except Exception as e:
        print(e)
        pickle.dump((size, initial, scaling, nl_func, dropout, optim, lrate, batch_size), open('loldraftassist/pro/test_dump.pkl', 'wb'))


sizes = [2, 3, 4, 5]
initials = [.5, .75, 1, 5, 10]
scalings = [.1, .25, .33, .5]
nl_funcs = [nn.ReLU, nn.Sigmoid]
dropouts = [0, .25, .5, .75]
optims = [optim.Adam, optim.AdamW, optim.Adagrad]
lrates = [1e-3, 1e-4, 1e-5]
batch_sizes = [25, 50, 100]


if __name__ == '__main__':
    fullX, fullY = db_to_tensor('db/2021pro.db', ('11.02', '11.03'))
    test_archs(sizes, initials, scalings, nl_funcs, dropouts, optims, lrates, batch_sizes, fullX, fullY)
