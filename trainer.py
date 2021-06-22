from argparse import ArgumentParser
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import warnings
import utils
warnings.filterwarnings("ignore")


class CycleGANTrainer(nn.Module):
    def __init__(self, my_model, my_logger, args):
        super().__init__()
        # save hyperparameters
        self.lr = args.learning_rate
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.lambda_ = args.loss_lambda

        # logger
        self.logger = my_logger
        self.log_img_ever_n_epoch = args.log_img_ever_n_epoch

        # device
        self.device = args.device

        # save the model
        self.disc_x = my_model['discriminator_x']
        self.disc_y = my_model['discriminator_y']
        self.gen_y2x = my_model['generator_y2x']
        self.gen_x2y = my_model['generator_x2y']

        # optimizer
        disc_param = list(self.disc_x.parameters()) + list(self.disc_y.parameters())
        self.optimizer_disc = torch.optim.Adam(disc_param, self.lr, (self.beta1, self.beta2))
        gen_param = list(self.gen_y2x.parameters()) + list(self.gen_x2y.parameters())
        self.optimizer_gen = torch.optim.Adam(gen_param, self.lr, (self.beta1, self.beta2))

    def train_model(self, data_module, epoch):
        # set epoch
        self.epoch = epoch
        # Switch model to training mode.
        self.model_to_train()
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

        # training loop
        for batch_idx, (batch_apple, batch_windows) in enumerate(pbar):
            """
            apple: x
            windows: y
            """

            # extract data from batch and put to device
            img_x = batch_apple[0].to(self.device)
            img_y = batch_windows[0].to(self.device)

            ##### DISCRIMINATOR ######
            # Reinitialize grad
            self.optimizer_disc.zero_grad()
            # loss disc real
            loss_disc_real_x = ((self.disc_x(img_x).squeeze() - 1) ** 2).mean()
            loss_disc_real_y = ((self.disc_y(img_y).squeeze() - 1) ** 2).mean()
            # loss disc fake
            loss_disc_fake_x = (self.disc_x(self.gen_y2x(img_y)) ** 2).mean()
            loss_disc_fake_y = (self.disc_y(self.gen_x2y(img_x)) ** 2).mean()
            # total discriminator loss
            loss_disc = 0.5 * (loss_disc_real_x + loss_disc_real_y + loss_disc_fake_x + loss_disc_fake_y)
            # Backward pass
            loss_disc.backward()
            # Optimize
            self.optimizer_disc.step()

            ##### Generator ######
            # Reinitialize grad
            self.optimizer_gen.zero_grad()
            # loss gan
            loss_gan_x = ((self.disc_x(self.gen_y2x(img_y)) - 1) ** 2).mean()
            loss_gan_y = ((self.disc_y(self.gen_x2y(img_x)) - 1) ** 2).mean()
            # loss cycle consistency
            loss_cycle_x = torch.abs_(self.gen_y2x(self.gen_x2y(img_x)) - img_x).mean()
            loss_cycle_y = torch.abs_(self.gen_x2y(self.gen_y2x(img_y)) - img_y).mean()
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

        if epoch % self.log_img_ever_n_epoch == 0 or epoch == 1:
            self.model_to_test()
            # Fixed test image to for sample visualization
            test_loader_apple, test_loader_windows = data_module.test_dataloader()
            test_x = next(iter(test_loader_apple))[0].to(self.device)
            test_y = next(iter(test_loader_windows))[0].to(self.device)
            # gen and save images
            folder_name = 'samples_during_training'
            img_names = ['epoch{}-app.png'.format(epoch), 'epoch{}-win.png'.format(epoch)]
            self.test_model_one(test_x, test_y, img_names, folder_name)
            self.model_to_train()

        # log losses
        mean_gen_loss = total_loss_gen / n_entries
        mean_disc_loss = total_loss_disc / n_entries
        self.logger.log('loss/gen_loss', mean_gen_loss, epoch)
        self.logger.log('loss/disc_loss', mean_disc_loss, epoch)

        return mean_disc_loss, mean_gen_loss

    def test_model_one(self, test_app, test_win, img_names, folder_name):
        # generate fake test images
        with torch.no_grad():
            test_app_fake = self.gen_x2y(test_app)
            test_win_fake = self.gen_y2x(test_win)
        # saves test image style transfer after each epoch
        self.logger.save_samples(test_app, test_app_fake, img_names[0], folder_name + '/apple')
        self.logger.save_samples(test_win, test_win_fake, img_names[1], folder_name + '/windows')

    def test_model(self, data_module):
        self.model_to_test()
        # get dataloaders
        test_loader_apple, test_loader_windows = data_module.test_dataloader(shuffle=True)
        # progress bar
        pbar = tqdm(zip(test_loader_apple, test_loader_windows),
                    desc='Testing',
                    leave=False,
                    total=min(len(test_loader_windows), len(test_loader_apple)))
        # test
        for batch_idx, (batch_apple, batch_windows) in enumerate(pbar):
            # extract data from batch and put to device
            img_app = batch_apple[0].to(self.device)
            img_win = batch_windows[0].to(self.device)

            # generate and save images
            folder_name = 'samples_after_training'
            img_names = ['testbatch{}-app.png'.format(batch_idx), 'testbatch{}-win.png'.format(batch_idx)]
            self.test_model_one(img_app, img_win, img_names, folder_name)

        pass

    def model_to_train(self):
        self.disc_x.train()
        self.disc_y.train()
        self.gen_y2x.train()
        self.gen_x2y.train()

    def model_to_test(self):
        self.disc_x.eval()
        self.disc_y.eval()
        self.gen_y2x.eval()
        self.gen_x2y.eval()

    def save_model(self, location, name):
        model = {
            'disc_y': [self.disc_y.state_dict()],
            'disc_x': [self.disc_x.state_dict()],
            'gen_y2x': [self.gen_y2x.state_dict()],
            'gen_x2y': [self.gen_x2y.state_dict()],
        }
        torch.save({'epoch': self.epoch,
                    'model': model,
                    },
                   os.path.join(location, name))

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--max_epochs', type=int, default=10,
                            help='maximum number of epochs (default: 10)')
        parser.add_argument('--learning_rate', type=float, default=3e-4,  # TODO
                            help='learning rate (default: 3e-4)')
        parser.add_argument('--beta1', type=float, default=0.5,  # TODO
                            help='beta 1 for adam (default: 0.5)')
        parser.add_argument('--beta2', type=float, default=0.999,  # TODO
                            help='beta 2 for adam (default: 0.999)')
        parser.add_argument('--loss_lambda', type=float, default=10,  # TODO
                            help='parameter for cycle consistency loss (default: 10)')
        parser.add_argument('--log_img_ever_n_epoch', type=int, default=10,  # TODO
                            help='log image ever n epoch (default: 10)')

        return parser
