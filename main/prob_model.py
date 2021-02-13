import torch as t
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import imageio
from main.ml import evaluate, db_to_tensor, shuffle_and_split, device
from collections import OrderedDict


conf_to_prob_arch = OrderedDict([('lin1', nn.Linear(1, 100)),
                                 ('sig1', nn.ReLU()),
                                 ('lin2', nn.Linear(100, 33)),
                                 ('relu2', nn.ReLU()),
                                 ('lin3', nn.Linear(33, 1)),
                                 ('batch', nn.BatchNorm1d(1))])
images = []


# train regression model of network confidence to estimated win probability
def gen_prob_model(arch, loss_fn, optimizer, xx, yy, iters=10000, lrate=7.5e-4):
    model = nn.Sequential(arch)
    if t.cuda.is_available():
        model = model.cuda()

    opt = optimizer(model.parameters(), lr=lrate, weight_decay=5e-5)
    fig, ax = plt.subplots()
    for i in range(iters):
        xt = t.tensor([xx]).T
        x_var = t.autograd.Variable(xt)
        yt = t.tensor([yy]).T
        y_var = t.autograd.Variable(yt)

        if t.cuda.is_available():
            out = model(x_var.cuda())
            loss = loss_fn(out, y_var.cuda())
        else:
            out = model(x_var)
            loss = loss_fn(out, y_var)

        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % 1000 == 0 and i != 0:
            print('Iteration: ', i)
        if i % 100 == 0:
            plt.cla()
            ax.set_title('Fitting Probability Model')
            ax.set_xlabel('Model Confidence')
            ax.set_ylabel('Estimated Probability')
            ax.plot(xx, yy, 'b.', label='Test Accuracies')
            ax.plot(xx, out.cpu().data.numpy(), 'r-', lw=3, label='Regression Model')
            ax.set_xlim(.475, 1.025)
            ax.set_ylim(.6175, .86625)
            ax.legend(loc=2)
            ax.text(0.55, .71, 'Step = %d' % i)
            ax.text(0.55, .7, 'Loss = %.4f' % loss.cpu().data.numpy())

            # Used to return the plot as an image array
            # (https://ndres.me/post/matplotlib-animated-gifs-easily/)
            fig.canvas.draw()  # draw the canvas, cache the renderer
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            images.append(image)

    return model.cpu()


def train_and_plot(patch):
    model = t.load(f'../assets/models/{patch}_model.pkl', map_location=device)
    valX, valY = pickle.load(open(f'../assets/models/{patch}_validation_data.pkl', 'rb'))
    confs = [.5 + i/100 for i in range(50)]
    vals = [evaluate(model, valX, valY, conf) for conf in confs]
    prob_model = gen_prob_model(conf_to_prob_arch, nn.MSELoss(), optim.Adam, confs, vals)
    prob_model.eval()
    model_outs = [prob_model(t.tensor([[conf]])) for conf in confs]
    plt.cla()
    plt.plot(confs, vals, 'b.', label='Test Accuracies')
    plt.plot(confs, model_outs, 'r-', label='Regression Model')
    plt.xlabel('Model Confidence')
    plt.ylabel('Estimated Probability')
    plt.title(f'Patch 11.3 Solo Queue Probability Model')  # TODO: better file naming system
    plt.legend(loc=2)
    plt.savefig(f'../graphics/{patch}_probability_model.png')
    plt.show()
    if input('Save?') == 'Y':
        imageio.mimsave(f'../graphics/fitting_prob_{patch}.gif', images, fps=10)
        t.save(prob_model, f'../assets/models/{patch}_probability_model.pkl')


if __name__ == '__main__':
    current_patch = '11_3soloq'
    train_and_plot(current_patch)
