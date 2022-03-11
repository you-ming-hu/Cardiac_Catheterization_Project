import torch

class BaseMetrics:
    def __init__(self,loss,criterion):
        assert criterion in [max,min]
        self.criterion = criterion
        assert loss._reduce.lower() == 'none'
        assert loss._output_numpy
        self.loss = loss
        
        self.best_result = None
        self.acc_count = 0
        self.acc_value = 0
        
    def __str__(self):
        return str(self.loss)
        
    def update_state(self,predict,label):
        with torch.no_grad():
            batch_values = self.get_new_state(predict,label)
        self.acc_count += batch_values.size
        self.acc_value += batch_values.sum()
        
    def result(self):
        return self.acc_value / self.acc_count
        
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

    