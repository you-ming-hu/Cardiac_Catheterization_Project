import torch
import torch.nn.functional as F

class BaseLoss:
    def __init__(self,subject,weights,schedule,**params):
        self.subject = subject
        self.weights = weights
        self.schedule = schedule
        self.name = f'{self.subject}_{self.__class__.__name__}'+'('+','.join([f'{k}={v}'  for k,v in params.items()])+')'
        for k,v in params.items():
            setattr(self,k,v)
        
    def __call__(self,output,data):
        loss = self.call(output[self.subject],data[self.subject])
        weights = data[self.weights] if self.weights is not None else self.weights
        return self.calculate_weighted_loss(loss,weights)
    
    def call(self,output,label):
        raise NotImplementedError
        
    def calculate_weighted_loss(self,loss,weights):
        if weights is None:
            weights = torch.ones_like(loss,dtype=loss.dtype)
        acc_count = weights.sum(axis=0)
        acc_loss = (loss * weights).sum(axis=0)
        return acc_count, acc_loss
        
class DiceLoss(BaseLoss):
    def __init__(self,subject,weights,smoothing=1,schedule=None):
        super().__init__(subject,weights,schedule,smoothing=smoothing)
    
    def call(self,output,label):
        output = torch.sigmoid(output)
        intersection = (output * label).sum(axis=[1,2])
        loss = (2.*intersection + self.smoothing)/(output.sum(axis=[1,2]) + label.sum(axis=[1,2]) + self.smoothing)
        loss = 1 - loss
        return loss

class BaseBCELoss(BaseLoss):
    def __init__(self,subject,weights,schedule,reduce_dim):
        super().__init__(subject,weights,schedule)
        self.reduce_dim = reduce_dim
        
    def call(self,output,label):
        loss = F.binary_cross_entropy_with_logits(output,label,reduction='none')
        if self.reduce_dim is not None:
            loss = loss.mean(axis=self.reduce_dim)
        return loss
    
class BCELoss2D(BaseBCELoss):
    def __init__(self,subject,weights,schedule=None):
        super().__init__(subject,weights,schedule,[1,2])
    
class BCELoss1D(BaseBCELoss):
    def __init__(self,subject,weights,schedule=None):
        super().__init__(subject,weights,schedule,[1])
        
class BCELoss(BaseBCELoss):
    def __init__(self,subject,weights,schedule=None):
        super().__init__(subject,weights,schedule,None)

