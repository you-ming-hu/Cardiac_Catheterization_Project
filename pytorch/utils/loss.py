import torch
import numpy as np

class BaseLoss:
    def __init__(self,reduce='mean',output_numpy=False,**kwdarg):
        assert reduce.lower() in ['sum','mean','none']
        self._reduce = reduce
        self._output_numpy = output_numpy
        self._kwdarg = kwdarg
    
    def __str__(self):
        return self.__class__.__name__+'('+','.join([f'{k}={v}' for k,v in self._kwdarg.items()])+')'
    
    def __call__(self,predict,label):
        value = self.call(predict,label)
        assert len(value.shape) == 1
        if self._reduce.lower() != 'none':
            value = getattr(torch,self._reduce)(value,axis=0)
        if self._output_numpy:
            value = value.cpu().numpy()
        if isinstance(value,torch.Tensor):
            assert not torch.isnan(value).all()
        else:
            assert not np.isnan(value).all()
        # return value
        return {'loss':value}
    
class CompoundLoss:
    def __init__(self,reduce='mean',output_numpy=False):
        assert reduce.lower() in ['sum','mean','none']
        self._reduce = reduce
        self._output_numpy = output_numpy
        self._losses = []
    
    def __str__(self):
        return ' + '.join([f'{w}*{l}' for w,l in self._losses])
    
    def add_loss(self,weight,loss,**kwdarg):
        self._losses.append((weight,loss(reduce=self._reduce,output_numpy=self._output_numpy,**kwdarg)))
        
    def __call__(self,predict,label):
        output = {'loss':0}
        for weight,loss_fn in self._losses:
            value = loss_fn(predict,label)['loss']
            output[str(loss_fn)] = value
            output['loss'] = output['loss'] + weight * value
        return output
        