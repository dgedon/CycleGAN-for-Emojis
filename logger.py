import os
import numpy as np
from matplotlib import pyplot as plt
import imageio
from torch.utils.tensorboard import SummaryWriter


class CycleGANLogger(object):
    def __init__(self, log_dir, args):
        self.log_dir = log_dir
        # init tensorboard
        self.logger = SummaryWriter(log_dir=self.log_dir)

    def log(self, name, val, iteration):
        self.logger.add_scalar(name, val, iteration)

    def plot_loss(self, loss_gen, loss_disc):
        plt.plot(loss_gen, label='Gen')
        plt.plot(loss_disc, label='Disc')
        plt.title('Loss During Training')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid()
        plt.legend()
        #plt.yscale('log')
        path = os.path.join(os.getcwd(), self.log_dir, 'plot_loss.png')
        plt.savefig(path)

    # Assignment code to produce before and after image grid
    def merge_images(self, sources, targets, batch_size):
        # shape: (batch_size, num_channels, h, w)
        _, _, h, w = sources.shape
        rows = int(np.sqrt(batch_size))
        sample_grid = np.zeros([3, rows * w, rows * w * 2])
        for idx, (s, t) in enumerate(zip(sources, targets)):
            if idx >= rows ** 2:
                break
            i = idx // rows
            j = idx % rows
            sample_grid[:, i * h:(i + 1) * h, (j * 2) * h:(j * 2 + 1) * h] = s
            sample_grid[:, i * h:(i + 1) * h, (j * 2 + 1) * h:(j * 2 + 2) * h] = t
        return sample_grid.transpose((1, 2, 0))  # shape: (w, h, c)

    def save_samples(self, img_true, img_fake, img_name, folder_name):
        img_true = img_true.cpu().detach().numpy()
        img_fake = img_fake.cpu().detach().numpy()
        batch_size = img_true.shape[0]

        # generate grid of images
        sample_grid_win_fake = self.merge_images(img_true, img_fake, batch_size)

        # get path to save
        save_path = os.path.join(self.log_dir, folder_name)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        path = os.path.join(save_path, img_name)

        # save images
        imageio.imwrite(path, sample_grid_win_fake)
