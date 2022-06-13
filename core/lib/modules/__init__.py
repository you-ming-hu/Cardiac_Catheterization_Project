import torch
                
class BinarySegHead(torch.nn.Sequential):
    def __init__(self,in_channels):
        super().__init__(torch.nn.Conv2d(in_channels,1,1,bias=False))
        
    def forward(self,x):
        x = super().forward(x)
        x = torch.squeeze(x,dim=1)
        return x
    
class BinaryClassifyHead(torch.nn.Sequential):
    def __init__(self,in_channels):
        super().__init__(torch.nn.Linear(in_channels,1,bias=False))
        
    def forward(self,x):
        x = super().forward(x)
        x = torch.squeeze(x,dim=1)
        return x