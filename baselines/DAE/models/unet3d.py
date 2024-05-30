import torch
import torch.nn as nn

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        self.encoder = None
        self.middle = None
        self.decoder = None

    def forward(self, x):
        pass

