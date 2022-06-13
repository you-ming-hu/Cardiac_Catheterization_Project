import torch
from core.lib.modules import BinaryClassifyHead

def create_module(in_channels):
    return torch.nn.Sequential(
            torch.nn.Linear(in_channels,in_channels//2,1),
            torch.nn.Mish(),
            BinaryClassifyHead(in_channels//2))