import torch

class BaseBackbone(torch.nn.Module):
    def __init__(self,stem, non_frozen_layers):
        super().__init__()
        self.stem = stem
        self.non_frozen_layers = non_frozen_layers
    
    def freeze(self):
        for param in self.stem.parameters():
            param.requires_grad = False
        
        for non_frozen_layer in self.non_frozen_layers:
            for param in non_frozen_layer.parameters():
                param.requires_grad = True
                
class BinarySegHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,x):
        if len(x.shape) == 4:
            x = torch.squeeze(x,dim=1)
        assert len(x.shape) == 3
        x = torch.sigmoid(x)
        return x

