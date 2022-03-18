from utils.model import BinarySegHead
import torch

class V1(BinarySegHead):
    def __init__(self,logit_output,in_channels):
        super().__init__(logit_output=logit_output)
        self.body = torch.nn.Sequential(
            # torch.nn.Conv2d(in_channels,in_channels*2,1),
            # torch.nn.Mish(),
            # torch.nn.Conv2d(in_channels*2,in_channels,1),
            # torch.nn.Mish(),
            torch.nn.Conv2d(in_channels,in_channels//2,1),
            torch.nn.Mish(),
            torch.nn.Conv2d(in_channels//2,1,1))
        
    def forward(self,x):
        x = self.body(x)
        x = super().forward(x)
        return x