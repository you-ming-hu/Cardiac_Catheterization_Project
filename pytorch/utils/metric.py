import torch
import numpy as np

class BaseMetrics:
    def __init__(self,criterion,loss_class,**kwdarg):
        assert criterion in [max,min]
        self.criterion = criterion
        self.loss = loss_class(reduce='none', output_numpy=True, **kwdarg)
        
        self.best_result = None
        self.acc_count = 0
        self.acc_value = 0
        
    def __str__(self):
        return str(self.loss)
        
    def update_state(self,predict,label):
        with torch.no_grad():
            batch_values = self.loss(predict,label)
        self.acc_count += batch_values.size
        self.acc_value += batch_values.sum()
        
    def result(self):
        value = self.acc_value / self.acc_count
        if isinstance(value,torch.Tensor):
            assert not torch.isnan(value).all()
        else:
            assert not np.isnan(value).all()
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
        self.acc_value = 0

    