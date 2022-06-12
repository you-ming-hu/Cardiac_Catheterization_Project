import torch
from core.lib.modules import BinarySegHead

def create_module(in_channels):
    return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels,in_channels//2,1),
            torch.nn.Mish(),
            BinarySegHead(in_channels//2))