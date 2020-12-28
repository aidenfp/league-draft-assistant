import torch as t
import matplotlib as plt
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import imageio
from collections import OrderedDict


conf_to_prob_arch = OrderedDict([('lin1', nn.Linear(1, 25)),
                                 ('sig1', nn.ReLU()),
                                 ('lin2', nn.Linear(25, 1)),
                                 ('batch', nn.BatchNorm1d(1))])
images = []


# train regression model of network confidence to estimated win probability
def gen_prob_model(arch, loss_fn, optimizer, xx, yy, iters=5000, lrate=1e-3):
    model = nn.Sequential(arch).cuda()
    opt = optimizer(model.parameters(), lr=lrate, momentum=.5, weight_decay=.0005)
    fig, ax = plt.subplots()
    for i in range(iters):
        xt = t.tensor([xx]).T
        x_var = t.autograd.Variable(xt)
        yt = t.tensor([yy]).T
        y_var = t.autograd.Variable(yt)
        out = model(x_var.cuda())
        loss = loss_fn(out, y_var.cuda())
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

    imageio.mimsave('fitting_prob.gif', images, fps=10)
    return model.cpu()


if __name__ == '__main__':
    confs = pickle.load(open('../assets/pkl/confs.pkl', "rb"))
    vals = pickle.load(open('../assets/pkl/vals.pkl', "rb"))
    prob_model = gen_prob_model(conf_to_prob_arch, nn.MSELoss(), optim.SGD, confs, vals)
    prob_model.eval()
    model_outs = [prob_model(t.tensor([[conf]])) for conf in confs]
    plt.cla()
    plt.plot(confs, vals, 'b.', label='Test Accuracies')
    plt.plot(confs, model_outs, 'r-', label='Regression Model')
    plt.xlabel('Model Confidence')
    plt.ylabel('Estimated Probability')
    plt.title('Fitting Probability Model')
    plt.legend(loc=2)
    plt.savefig('probmodel.png')
    plt.show()
    if input('Save?') == 'Y':
        t.save(prob_model, 'probability_model.pkl')