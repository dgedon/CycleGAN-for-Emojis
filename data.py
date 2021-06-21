import os
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


class EmojiDatamodule(Dataset):
    def __init__(self, args):
        # save the hyperparameters
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.img_size = args.img_size

        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_path_apple = os.path.join('./dataset', 'Apple')
        test_path_apple = os.path.join('./dataset', 'Test_Apple')
        train_path_win = os.path.join('./dataset', 'Windows')
        test_path_win = os.path.join('./dataset', 'Test_Windows')

        self.train_dataset_apple = datasets.ImageFolder(train_path_apple, transform)
        self.test_dataset_apple = datasets.ImageFolder(test_path_apple, transform)
        self.train_dataset_win = datasets.ImageFolder(train_path_win, transform)
        self.test_dataset_win = datasets.ImageFolder(test_path_win, transform)

    def train_dataloader(self, shuffle=True):
        dl_apple = DataLoader(
            self.train_dataset_apple,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )
        dl_windows = DataLoader(
            self.train_dataset_win,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )
        return dl_apple, dl_windows

    def test_dataloader(self, shuffle=False):
        dl_apple = DataLoader(
            self.test_dataset_apple,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )
        dl_windows = DataLoader(
            self.test_dataset_win,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )
        return dl_apple, dl_windows

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--img_size', type=int, default=32,
                            help='image size rescaling (default: 32)')
        parser.add_argument('--batch_size', type=int, default=32,
                            help='train batch size (default: 256)')
        parser.add_argument('--num_workers', type=int, default=0,
                            help='number of workers for dataloader (default: 4)')

        return parser
