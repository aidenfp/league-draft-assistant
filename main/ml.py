import numpy as np
import torch as t
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sqlite3 as sql
import matplotlib.pyplot as plt
import pickle
from collections import OrderedDict
from main.data import num_champs, db_to_tensor


device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
print('Device: ', device)
if device == t.device("cuda:0"):
    t.cuda.device(device)

model_arch = OrderedDict([('lin1', nn.Linear(2 * num_champs, 232)),
                        ('relu1', nn.ReLU()),
                        ('lin2', nn.Linear(232, 2)),
                        ('batchnorm', nn.BatchNorm1d(2))])


# To train the model with champion input vectors and win confidence outputs,
# used the following parameters:
# optimizer=optim.Adam      lrate=1.5e-3        k=100       loss_fn=nn.CrossEntropyLoss()
# iters=2500        network arch=model_arch
def batch_gd(network, loss_fn, optimizer, x, y, iters, lrate=1.5e-3, k=100):
    n, d = tuple(x.size())

    opt = optimizer(network.parameters(), lr=lrate)

    if t.cuda.is_available():
        network = network.cuda()
    np.random.seed(0)
    num_updates = 0
    indices = np.arange(n)
    while num_updates < iters:

        np.random.shuffle(indices)
        x = x[indices, :]
        y = y[indices, :]

        xt = x[:k, :].float()
        yt = y[:k, :].reshape((k,)).long()
        if t.cuda.is_available():
            xt = xt.cuda()
            yt = yt.cuda()

        out = network(xt)
        loss = loss_fn(out, yt)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if num_updates % 100 == 0:
            print(num_updates)

        num_updates += 1


def evaluate(network, x, y, conf_thresh=None):
    n, d = tuple(x.size())
    correct = 0
    count = 0
    network.eval()
    with t.no_grad():
        for i in range(n):
            xt = x[i:i + 1, :]
            yt = y[i:i + 1, :]

            if t.cuda.is_available():
                xt = xt.cuda()
                yt = yt.cuda()
            out = F.softmax(network(xt.float()), dim=1)
            pred = t.argmax(out)
            conf = t.max(out).cpu().numpy()
            # only checks accuracy of "confident" predictions
            if conf_thresh:
                if conf >= conf_thresh:
                    if int(pred) == int(yt): correct += 1
                    count += 1
                else:
                    continue
            else:
                if int(pred) == int(yt): correct += 1
                count += 1
    return correct / count


def cross_val(arch, x, y, k=10, lam=1.5e-3, b_size=100):
    n, d = tuple(x.size())
    x_split = list(t.split(x, n // k, dim=1))
    y_split = list(t.split(y, n // k, dim=1))
    network = nn.Sequential(arch)
    scores = []
    for i in range(k):
        # fold and split
        x_copy = x_split.copy()
        y_copy = y_split.copy()
        x_i = x_copy.pop(i)
        y_i = y_copy.pop(i)
        x_minus_i = t.hstack(tuple([tensor for tensor in x_copy]))
        y_minus_i = t.hstack(tuple([tensor for tensor in y_copy]))

        # reset our networks weights for each fold
        for layer in network.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        # update network weights and score on data left out
        network.train()
        batch_gd(network, nn.CrossEntropyLoss(), optim.Adam, x_minus_i, y_minus_i, 1000, lam, b_size)
        score = evaluate(network, x_i, y_i)
        print('Score of i={}: {}'.format(i, score))
        scores.append(score)
    return sum(scores) / len(scores)


def shuffle_and_split(x, y, sizes):
    n, d = tuple(x.size())

    # shuffle
    np.random.seed(0)
    indices = np.arange(n)
    np.random.shuffle(indices)
    x = x[indices, :]
    y = y[indices, :]

    # split
    out = []
    prev = 0
    for size in sizes:
        dx = x[prev:prev + size, :]
        dy = y[prev:prev + size, :]
        out.append((dx, dy))
        prev = prev + size
    return out


def train_and_evaluate(full_x, full_y, arch, path):
    splits = shuffle_and_split(full_x, full_y, [int(.9 * full_x.shape[0]), full_x.shape[0] - int(.9 * full_x.shape[0])])
    largeX, largeY = splits[0]
    valX, valY = splits[1]
    model = nn.Sequential(arch)
    batch_gd(model, nn.CrossEntropyLoss(), optim.Adam, largeX, largeY, 5000)
    print('Total evaluation: ', evaluate(model, valX, valY))
    if input("Save?") == 'Y':
        t.save(model, f'{path}_model.pkl')
        pickle.dump((valX, valY), open(f'{path}_validation_data.pkl', 'wb'))


if __name__ == '__main__':
    fullX, fullY = db_to_tensor(f'../db/11_3soloq.db')
    train_and_evaluate(fullX, fullY, model_arch, '11_3soloq')
