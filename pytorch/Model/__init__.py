import torch

from .backbone import torchvision, smp

class BaseModel(torch.nn.Module):
    def __init__(self,descriotion):
        super().__init__()
        self.hyperparam_dict
        
    def __str__(self):
        return '.'.join([__name__,self.__class__.__name__])+'('+','.join([f'{k}={v}' for k,v in self.hyperparam_dict.items()]) + ')'
        
    
class FirstModel():
    def __
    
    
