import torch
import numpy as np

class BaseMetrics:
    def __init__(self,criterion,loss_class,**kwdarg):
        assert criterion in [max,min]
        self.criterion = criterion
        self.loss_fn = loss_class(reduce='none', output_numpy=True, **kwdarg)
        
        self.best_result = None
        self.acc_count = 0
        # self.acc_loss = 0
        self.acc_values = {}
        
    def __str__(self):
        return str(self.loss_fn)
        
    def update_state(self,predict,label):
        with torch.no_grad():
            batch_values = self.loss_fn(predict,label)
            # loss = batch_values['loss']
            
        self.acc_count += batch_values['loss'].size
        for loss_name, loss_value in batch_values.items():
            try:
                self.acc_values[loss_name] += loss_value.sum()
            except KeyError:
                self.acc_values[loss_name] = loss_value.sum()
        
        # self.acc_loss += loss.sum()
        
    def result(self,detail=False):
        if detail:
            value = {loss_name : acc_loss_value/self.acc_count for loss_name,acc_loss_value in self.acc_values.items()}
        else:
            value = self.acc_values['loss'] / self.acc_count
        return value
        
    def update_best_result(self):
        if self.best_result == None:
            self.best_result = self.result()
            is_updated = True
        else:
            new_best_result = self.criterion(self.best_result,self.result())
            if new_best_result != self.best_result:
                self.best_result = new_best_result
                is_updated = True
            else:
                is_updated = False
        return is_updated
        
    def reset_state(self):
        self.acc_count = 0
        # self.acc_value = 0
        for loss_name in self.acc_values.keys():
            self.acc_values[loss_name] = 0

    