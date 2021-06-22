import torch
import torch.nn as nn


def get_model(args):
    my_dict = {}
    # discriminator
    my_dict['discriminator_x'] = Discriminator().to(args.device)
    my_dict['discriminator_y'] = Discriminator().to(args.device)

    # generators
    my_dict['generator_x'] = Generator().to(args.device)
    my_dict['generator_y'] = Generator().to(args.device)

    return my_dict


class Generator(nn.Module):
    """Defines the architecture of the generator network.
       Note: Both generators G_XtoY and G_YtoX have the same architecture in this assignment.
    """
    def __init__(self):
        super(Generator, self).__init__()
        self.kernel_size = 4

        # encoder
        self.conv1 = conv(in_channels=3, out_channels=32, kernel_size=self.kernel_size)
        self.conv2 = conv(in_channels=32, out_channels=64, kernel_size=self.kernel_size)

        # transformation part
        self.resnet_block = ResnetBlock(conv_dim=64)

        # decoder
        self.deconv1 = deconv(in_channels=64, out_channels=32, kernel_size=self.kernel_size, stride=2, padding=1)
        self.deconv2 = deconv(in_channels=32, out_channels=3, kernel_size=self.kernel_size, stride=2, padding=1)

        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))

        out = self.relu(self.resnet_block(out))

        out = self.relu(self.deconv1(out))
        out = self.tanh(self.deconv2(out))

        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.kernel_size = 4

        self.conv1 = conv(in_channels=3, out_channels=32, kernel_size=self.kernel_size)
        self.conv2 = conv(in_channels=32, out_channels=64, kernel_size=self.kernel_size)
        self.conv3 = conv(in_channels=64, out_channels=128, kernel_size=self.kernel_size)
        self.conv4 = conv(in_channels=128, out_channels=1, kernel_size=self.kernel_size, padding=0)

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):

        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))

        out = self.conv4(out).squeeze()
        out = self.sigmoid(out)
        return out


class ResnetBlock(nn.Module):
    def __init__(self, conv_dim):
        super(ResnetBlock, self).__init__()
        self.conv_layer = conv(in_channels=conv_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = x + self.conv_layer(x)
        return out


def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1):
    conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    bn = nn.BatchNorm2d(out_channels)

    return nn.Sequential(conv, bn)


def conv(in_channels, out_channels, kernel_size, stride=2, padding=1):
    conv_layer = nn.Conv2d(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=padding,
                           bias=False)
    bn = nn.BatchNorm2d(out_channels)

    return nn.Sequential(conv_layer, bn)
