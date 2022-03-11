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
                
class BaseHead(torch.nn.Module):
    pass

