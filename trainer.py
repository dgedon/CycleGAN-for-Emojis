from argparse import ArgumentParser
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import warnings
import utils
warnings.filterwarnings("ignore")


class CycleGANTrainer(nn.Module):
    def __init__(self, my_model, args):
        super().__init__()
        # save hyperparameters
        self.lr = args.lr
        self.lambda_ = 1

        # save the model
        self.disc_y2x = my_model['discriminator_x']
        self.disc_x2y = my_model['discriminator_y']
        self.gen_x = my_model['generator_x']
        self.gen_y = my_model['generator_y']

        # optimizer
        disc_param = list(self.disc_y2x.parameters()) + list(self.disc_x2y.parameters())
        self.optimizer_disc = torch.optim.Adam(disc_param, self.lr)
        gen_param = list(self.gen_x.parameters()) + list(self.gen_y.parameters())
        self.optimizer_gen = torch.optim.Adam(gen_param, self.lr)

    def train_model(self, args, data_module, epoch, log_dir):
        # set epoch
        self.epoch = epoch
        # Switch model to training mode.
        self.disc_y2x.train()
        self.disc_x2y.train()
        self.gen_x.train()
        self.gen_y.train()

        # allocate
        total_loss_disc = 0
        total_loss_gen = 0
        n_entries = 0

        # get dataloaders
        train_loader_apple, train_loader_windows = data_module.train_dataloader(shuffle=True)
        # progress bar
        pbar = tqdm(zip(train_loader_apple, train_loader_windows),
                    desc='Training epoch {}'.format(epoch),
                    leave=False,
                    total=min(len(train_loader_windows), len(train_loader_apple)))

        # Fixed test image to for sample visualization
        test_loader_apple, test_loader_windows = data_module.test_dataloader()
        test_X = next(iter(test_loader_apple))[0]
        test_Y = next(iter(test_loader_windows))[0]

        # training loop
        for batch_idx, (batch_apple, batch_windows) in enumerate(pbar):
            """
            apple: x
            windows: y
            """

            # extract data from batch and put to device
            img_x = batch_apple[0].to(args.device)
            img_y = batch_windows[0].to(args.device)

            ##### DISCRIMINATOR ######
            # Reinitialize grad
            self.optimizer_disc.zero_grad()
            # loss disc real
            loss_disc_real_x = ((self.disc_y2x(img_x).squeeze() - 1) ** 2).mean()
            loss_disc_real_y = ((self.disc_x2y(img_y).squeeze() - 1) ** 2).mean()
            # loss disc fake
            loss_disc_fake_x = (self.disc_y2x(self.gen_x(img_y)) ** 2).mean()
            loss_disc_fake_y = (self.disc_x2y(self.gen_y(img_x)) ** 2).mean()
            # total discriminator loss
            loss_disc = loss_disc_real_x + loss_disc_real_y + loss_disc_fake_x + loss_disc_fake_y
            # Backward pass
            loss_disc.backward()
            # Optimize
            self.optimizer_disc.step()

            ##### Generator ######
            # Reinitialize grad
            self.optimizer_gen.zero_grad()
            # loss gan
            loss_gan_x = ((self.disc_y2x(self.gen_x(img_y)) - 1) ** 2).mean()
            loss_gan_y = ((self.disc_x2y(self.gen_y(img_x)) - 1) ** 2).mean()
            # loss cycle consistency
            loss_cycle_x = torch.abs_(self.gen_x(self.gen_y(img_x)) - img_x).mean()
            loss_cycle_y = torch.abs_(self.gen_y(self.gen_x(img_y)) - img_y).mean()
            # total generator loss
            loss_gen = loss_gan_x + loss_gan_y + self.lambda_ * (loss_cycle_x + loss_cycle_y)
            # Backward pass
            loss_gen.backward()
            # Optimize
            self.optimizer_gen.step()

            # Update progress bar
            bs = img_x.size(0)
            n_entries += bs
            total_loss_disc += loss_disc.detach().cpu().numpy()
            total_loss_gen += loss_gen.detach().cpu().numpy()
            pbar.set_postfix({
                'disc_loss': total_loss_disc / n_entries,
                'gen_loss': total_loss_gen / n_entries
            })

        # Saves test image style transfer after each epoch
        utils.save_samples(test_X, test_Y, self.gen_y, self.gen_x, log_dir, epoch, args)

        return total_loss_disc / n_entries, total_loss_gen / n_entries

    def save_model(self, location, name):
        model = {
            'disc_x2y': self.disc_x2y.state_dict(),
            'disc_y2x': self.disc_y2x.state_dict(),
            'gen_x': self.gen_x.state_dict(),
            'gen_y': self.gen_y.state_dict()
        }
        torch.save({'epoch': self.epoch,
                    'model': [self.model.state_dict()],
                    },
                   os.path.join(location, name))

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--max_epochs', type=int, default=10,
                            help='maximum number of epochs (default: 10)')
        parser.add_argument('--lr', type=int, default=1e-3,  # TODO
                            help='learning rate (default: 1e-3)')

        return parser
