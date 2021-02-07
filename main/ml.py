import numpy as np
import torch as t
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sqlite3 as sql
# import matplotlib.pyplot as plt
import pickle
from collections import OrderedDict
from data import num_champs, champs


#available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
#print(available_gpus)
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
    d, n = tuple(x.size())

    opt = optimizer(network.parameters(), lr=lrate)

    network = network.cuda()
    np.random.seed(0)
    num_updates = 0
    indices = np.arange(n)
    while num_updates < iters:

        np.random.shuffle(indices)
        x = x[:, indices]
        y = y[:, indices]
        if num_updates >= iters: break

        xt = x[:, :k].float()
        xt = xt.cuda()
        yt = y[:, :k][0].long()
        yt = yt.cuda()

        out = network(xt.T)
        loss = loss_fn(out, yt)

        opt.zero_grad()
        loss.backward()
        opt.step()

        num_updates += 1


def evaluate(network, x, y, conf_thresh=None):
    d, n = tuple(x.size())
    correct = 0
    count = 0
    network.eval()
    with t.no_grad():
        for i in range(n):
            xt = x[:, i:i + 1]
            xt = xt.cuda()
            yt = y[:, i:i + 1]
            yt = yt.cuda()
            out = F.softmax(network(xt.T.float()))
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
    d, n = tuple(x.size())
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


# Converts list of champion names into one-hot encoded vector
def champs_to_vec(champ_list):
    return np.isin(champs, champ_list).astype(dtype=np.uint8)


# takes in column vector of 2*num_champs and returns two lists corresponding to the selected champions for each team
# INCOMPLETE
def vec_to_champs(vector):
    blue = vector[:num_champs, :]
    red = vector[num_champs:, :]
    blue_inds = np.argmax()
    red_inds = np.argmax()


# pull match data from database and converts to torch tensors
def db_to_tensor(db):
    conn = sql.connect(db)
    c = conn.cursor()
    matches = c.execute('''SELECT * FROM matches_table;''').fetchall()
    x = np.zeros((2 * num_champs, len(matches)))
    y = np.zeros((1, len(matches)))
    i = 0
    for match in matches:
        blue = [match[i] for i in range(5)]
        red = [match[i + 5] for i in range(5)]
        win = match[10]
        blue_vec = champs_to_vec(blue)
        red_vec = champs_to_vec(red)
        x_vec = np.vstack((blue_vec, red_vec))
        x[:, i:i + 1] = x_vec
        y[:, i:i + 1] = 1 if win == 'B' else 0
        i += 1
    return t.from_numpy(x), t.from_numpy(y)


def shuffle_and_split(x, y, sizes):
    d, n = tuple(x.size())

    # shuffle
    np.random.seed(0)
    indices = np.arange(n)
    np.random.shuffle(indices)
    x = x[:, indices]
    y = y[:, indices]

    # split
    out = []
    prev = 0
    for size in sizes:
        dx = x[:, prev:prev + size]
        dy = y[:, prev:prev + size]
        out.append((dx, dy))
        prev = prev + size
    return out


def train_and_evaluate(arch, db):
    fullX, fullY = db_to_tensor('../db/' + db + '.db')
    splits = shuffle_and_split(fullX, fullY, [9000, 1000])
    largeX, largeY = splits[0]
    valX, valY = splits[1]
    model = nn.Sequential(arch)
    batch_gd(model, nn.CrossEntropyLoss(), optim.Adam, largeX, largeY, 2500)
    print('Total evaluation: ', evaluate(model, valX, valY))
    if input("Save?") == 'Y':
        t.save(model, f'../assets/pkl/{db}_model.pkl')


if __name__ == '__main__':
    train_and_evaluate(model_arch, '11_3soloq')
