import argparse
import torch
import random
from tqdm import tqdm
import os

import utils
from data import EmojiDatamodule
from model import get_model
from trainer import CycleGANTrainer
from logger import CycleGANLogger

if __name__ == '__main__':
    # argparser
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for number generator (default: 42)')
    parser.add_argument('--save_model', action='store_true',
                        help='if activated, stores the last model')

    # get arguments from datamodule and trainer
    parser = EmojiDatamodule.add_model_specific_args(parser)
    parser = CycleGANTrainer.add_model_specific_args(parser)
    args = parser.parse_args()

    # create logging folder
    log_dir = utils.set_log_dir()
    # save the config file
    utils.save_config(log_dir, args)

    # set seed
    utils.seed_everything(args.seed)

    # Set device
    use_cuda = torch.cuda.is_available()
    args.device = torch.device('cuda:0' if use_cuda else 'cpu')
    tqdm.write("Using {}!\n".format(args.device))

    # get dataloaders
    tqdm.write("Define dataloaders...")
    data_module = EmojiDatamodule(args)
    tqdm.write("Done!\n")

    # load model
    tqdm.write('Define model(s)...')
    my_models = get_model(args)
    tqdm.write('Done!\n')

    # define logger
    tqdm.write('Define logger...')
    my_logger = CycleGANLogger(log_dir, args)
    tqdm.write('Done!\n')

    # set up Trainer
    tqdm.write("Set up trainer...")
    trainer = CycleGANTrainer(my_models, my_logger, args)
    tqdm.write("Done!\n")

    # training
    # allocation
    loss_disc_list = []
    loss_gen_list = []
    tqdm.write("Start training.")
    for epoch in range(1, args.max_epochs + 1):
        # training epoch
        loss_disc, loss_gen = trainer.train_model(data_module, epoch)

        # save loss
        loss_gen_list.append(loss_gen)
        loss_disc_list.append(loss_disc)

        # save the last model
        if args.save_model and epoch == args.max_epochs + 1:
            trainer.save_model(log_dir, 'model_last.pth')
    tqdm.write("Training finished.\n")

    # generate loss curve and save
    my_logger.plot_loss(loss_gen_list, loss_disc_list)

    # generate test images
    tqdm.write("Start testing.")
    trainer.test_model(data_module)
    tqdm.write("Testing done.")
