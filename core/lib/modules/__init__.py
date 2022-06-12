import torch
                
class BinarySegHead(torch.nn.Sequential):
    def __init__(self,in_channels):
        super().__init__(torch.nn.Conv2d(in_channels,1,1))
        
    def forward(self,x):
        x = super()(x)
        x = torch.squeeze(x,dim=1)
        return x