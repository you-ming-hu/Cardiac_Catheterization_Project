class Base:
    def __str__(self):
        return '.'.join([__name__,self.__class__.__name__])+'('+','.join([f'{k}={v}' for k,v in vars(self).items()]) + ')'

class Constant(Base):
    def __init__(self,p):
        self.p = p
    def __call__(self,epoch):
        return self.p
    
class WarmUp(Base):
    def __init__(self,start_p,max_p,warmup_epochs):
        self.start_p = start_p
        self.max_p = max_p
        self.warmup_epochs = warmup_epochs
        self.slope = (max_p - start_p)/warmup_epochs
    
    def __call__(self,epoch):
        if epoch < self.warmup_epochs:
            p = self.start_p + self.slope * epoch
        else:
            p = self.max_p = self.max_p
        return p