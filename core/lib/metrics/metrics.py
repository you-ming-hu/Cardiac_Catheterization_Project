import torch
import torch.nn.functional as F

class BaseMetric:
    def __init__(self,subject,weights,**params):
        self.subject = subject
        self.weights = weights
        self.name = f'{self.subject}_{self.__class__.__name__}'+'('+','.join([f'{k}={v}'  for k,v in params.items()])+')'
        for k,v in params.items():
            setattr(self,k,v)
        
    def __call__(self,output,data):
        with torch.no_grad():
            metric = self.call(output[self.subject], data[self.subject])
            weights = data[self.weights] if self.weights is not None else self.weights
            acc_count, acc_metric = self.calculate_weighted_metric(metric,weights)
        return acc_count, acc_metric
    
    def calculate_weighted_metric(self,metric,weights):
        if weights is None:
            weights = torch.ones_like(metric,dtype=metric.dtype)
        acc_count = weights.sum(axis=0)
        acc_metric = (metric * weights).sum(axis=0)
        return acc_count, acc_metric
        
    def call(self):
        raise NotImplementedError

class DiceLoss(BaseMetric):
    def __init__(self,subject,weights,smoothing=1):
        super().__init__(subject,weights,smoothing=smoothing)
        
    def call(self,output,label):
        output = torch.sigmoid(output)
        intersection = (output * label).sum(axis=[1,2])
        metric = (2.*intersection + self.smoothing)/(output.sum(axis=[1,2]) + label.sum(axis=[1,2]) + self.smoothing)
        metric = 1 - metric
        return metric

class DiceAccuracy(BaseMetric):
    def __init__(self,subject,weights,threshold=0.5,smoothing=1):
        super().__init__(subject,weights,threshold=threshold,smoothing=smoothing)
    
    def call(self,output,label):
        output = torch.sigmoid(output)
        output = (output >= self.threshold).astype(label.dtype)
        intersection = (output * label).sum(axis=[1,2])
        metric = (2.*intersection + self.smoothing)/(output.sum(axis=[1,2]) + label.sum(axis=[1,2]) + self.smoothing)
        return metric

class BaseBCELoss(BaseMetric):
    def __init__(self,subject,weights,reduce_dim):
        super().__init__(subject,weights)
        self.reduce_dim = reduce_dim
        
    def call(self,output,label):
        metric = F.binary_cross_entropy_with_logits(output,label,reduction='none')
        if self.reduce_dim is not None:
            metric = metric.mean(axis=self.reduce_dim)
        return metric
    
class BCELoss(BaseBCELoss):
    def __init__(self,subject,weights):
        super().__init__(subject,weights,None)
        
class BCELoss1D(BaseBCELoss):
    def __init__(self,subject,weights):
        super().__init__(subject,weights,[1])
        
class BCELoss2D(BaseBCELoss):
    def __init__(self,subject,weights):
        super().__init__(subject,weights,[1,2])

class BaseBinaryAccuracy(BaseMetric):
    def __init__(self,subject,weights,threshold,reduce_dim):
        super().__init__(subject,weights,threshold=threshold)
        self.reduce_dim = reduce_dim
        
    def call(self,output,label):
        output = torch.sigmoid(output)
        output = output >= self.threshold
        metric = output == label
        metric = metric.astype(float)
        if self.reduce_dim is not None:
            metric = metric.mean(axis=self.reduce_dim)
        return metric
    
class BinaryAccuracy(BaseBinaryAccuracy):
    def __init__(self,subject,weights,threshold):
        super().__init__(subject,weights,threshold,None)
        
class BinaryAccuracy1D(BaseBinaryAccuracy):
    def __init__(self,subject,weights,threshold):
        super().__init__(subject,weights,threshold,[1])
        
class BinaryAccuracy2D(BaseBinaryAccuracy):
    def __init__(self,subject,weights,threshold):
        super().__init__(subject,weights,threshold,[1,2])