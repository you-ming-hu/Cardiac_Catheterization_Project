import torch
import numpy as np

class BaseMetrics:
    def __init__(self,loss_class,**kwdarg):
        self.loss_fn = loss_class(reduce='none', output_numpy=True, **kwdarg)
        
        self.acc_count = 0
        self.acc_values = {}
        
    def __str__(self):
        return str(self.loss_fn)
        
    def update_state(self,predict,label):
        with torch.no_grad():
            batch_values = self.loss_fn(predict,label)
            
        self.acc_count += batch_values['loss'].size
        for loss_name, loss_value in batch_values.items():
            try:
                self.acc_values[loss_name] += loss_value.sum()
            except KeyError:
                self.acc_values[loss_name] = loss_value.sum()
        
    def result(self,detail=False):
        if detail:
            value = {loss_name : acc_loss_value/self.acc_count for loss_name,acc_loss_value in self.acc_values.items()}
        else:
            value = self.acc_values['loss'] / self.acc_count
        return value
        
    def reset_state(self):
        self.acc_count = 0
        for loss_name in self.acc_values.keys():
            self.acc_values[loss_name] = 0

    