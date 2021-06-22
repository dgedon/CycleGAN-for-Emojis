import os
import datetime
import torch
import numpy as np
import random
import json
import imageio
from matplotlib import pyplot as plt


def seed_everything(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


def set_log_dir():
    folder = os.path.join(os.getcwd(), 'logs', 'output_' +
                          str(datetime.datetime.now()).replace(":", "_").replace(" ", "_").replace(".", "_"))
    # Create output folder if needed
    try:
        os.makedirs(folder)
    except FileExistsError:
        pass
    return folder


def fname(folder, name, prefix=''):
    return os.path.join(folder, (prefix + '_' + name) if prefix else name)


def save_config(folder, args, prefix=''):
    with open(fname(folder, 'config.json', prefix), 'w') as f:
        json.dump(vars(args), f, indent='\t')


def plot_loss(loss_gen, loss_disc, folder):
    plt.plot(loss_gen, label='Gen')
    plt.plot(loss_disc, label='Disc')
    plt.title('Loss During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
    path = os.path.join(os.getcwd(), folder, 'plot_loss.png')
    plt.savefig(path)


# Assignment code to produce before and after image grid
def merge_images(sources, targets, args):
    # shape: (batch_size, num_channels, h, w)
    _, _, h, w = sources.shape
    rows = int(np.sqrt(args.batch_size))
    sample_grid = np.zeros([3, rows*w, rows*w*2])
    for idx, (s, t) in enumerate(zip(sources, targets)):
        i = idx // rows
        j = idx % rows
        sample_grid[:, i * h:(i + 1) * h, (j * 2) * h:(j * 2 + 1) * h] = s
        sample_grid[:, i * h:(i + 1) * h, (j * 2 + 1) * h:(j * 2 + 2) * h] = t
    return sample_grid.transpose(1, 2, 0)  # shape: (w, h, c)


def save_samples(X, Y, Gen_y, Gen_x, log_dir, epoch, args):
    Y_gen = Gen_y(X)
    X_gen = Gen_x(Y)

    X = X.cpu().detach().numpy()
    Y = Y.cpu().detach().numpy()
    Y_gen = Y_gen.cpu().detach().numpy()
    X_gen = X_gen.cpu().detach().numpy()

    sample_grid_Y_gen = merge_images(X, Y_gen, args)
    sample_grid_X_gen = merge_images(Y, X_gen, args)

    path_Y_gen = os.path.join(log_dir, 'sample-Y_gen-epoch-{}.png'.format(epoch))
    path_X_gen = os.path.join(log_dir, 'sample-X_gen-epoch-{}.png'.format(epoch))
    # im_Y_gen = Image.fromarray(np.uint8(sample_grid_Y_gen) * 255).convert('RGB')
    # im_X_gen = Image.fromarray(np.uint8(sample_grid_X_gen) * 255).convert('RGB')
    # im_Y_gen.save(path_Y_gen)
    # im_X_gen.save(path_X_gen)
    # TODO: Not sure whether we should fix the warning message.
    # This suppresses the warning but the images are very dark
    # imageio.imwrite(path_Y_gen, sample_grid_Y_gen.astype(np.uint8))
    # imageio.imwrite(path_X_gen, sample_grid_X_gen.astype(np.uint8))
    imageio.imwrite(path_Y_gen, sample_grid_Y_gen)
    imageio.imwrite(path_X_gen, sample_grid_X_gen)

    print('Samples saved')