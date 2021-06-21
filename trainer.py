from argparse import ArgumentParser
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os


class CycleGANTrainer(nn.Module):
    def __init__(self, my_model, args):
        super(CycleGANTrainer).__init__()
        # save hyperparameters
        self.lr = args.lr

        # save the model
        self.discrim_x = my_model
        """self.discrim_y = my_model[1]
        self.gen_x = my_model[2]
        self.gen_y = my_model[3]"""

        # optimizer
        disc_param = self.discrim_x.parameters()
        self.optimizer = torch.optim.Adam(disc_param, self.lr)

    def train_model(self, args, data_module, epoch):
        # set epoch
        self.epoch = epoch
        # Switch model to training mode.
        self.model.train()
        # allocate
        total_loss = 0
        n_entries = 0
        # get dataloaders
        train_loader = data_module.train_dataloader()
        # progress bar
        pbar = tqdm(train_loader, desc='Training epoch {}'.format(self.run_iter, epoch), leave=False)

        # training loop
        for batch_idx, batch in enumerate(pbar):
            # extract data from batch and put to device
            traces, labels, ids, age_sex = batch
            traces = traces.to(device=args.device)
            labels = labels.to(device=args.device)
            age_sex = age_sex.to(device=args.device)

            # Reinitialize grad
            self.model.zero_grad()
            # Forward pass
            inp = traces, age_sex
            logits = self.model(inp)
            # loss function
            train_loss = self.loss_fun(logits, labels)
            # Backward pass
            train_loss.backward()
            # Optimize
            self.optimizer.step()

            # Update
            total_loss += train_loss.detach().cpu().numpy()
            bs = labels.size(0)
            n_entries += bs
            pbar.set_postfix({'loss': total_loss / n_entries})

        # update schedule
        self.scheduler.step()

        # mean train loss
        mean_train_loss = total_loss / n_entries
        return mean_train_loss

    def save_model(self, location, name):
        torch.save({'epoch': self.epoch,
                    'model': self.model.state_dict(),
                    },
                   os.path.join(location, name))

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--max_epochs', type=int, default=200,
                            help='maximum number of epochs (default: 10)')
        parser.add_argument('--lr', type=int, default=1e-3,  # TODO
                            help='learning rate (default: 1e-3)')

        return parser
