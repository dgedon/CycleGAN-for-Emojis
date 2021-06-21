import os
import datetime
import torch
import numpy as np
import random
import json
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
