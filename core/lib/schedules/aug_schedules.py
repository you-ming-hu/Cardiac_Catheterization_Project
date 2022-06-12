class Base:
    def __call__(self,epoch):
        assert isinstance(epoch,int)
        return self.call(epoch)
    def call(self,epoch):
        raise NotImplementedError

class Constant(Base):
    def __init__(self,p):
        self.p = p
    def call(self,epoch):
        return self.p
    
class WarmUp(Base):
    def __init__(self,start_p,max_p,warmup_epochs):
        self.start_p = start_p
        self.max_p = max_p
        self.warmup_epochs = warmup_epochs
        self.slope = (max_p - start_p)/warmup_epochs
    
    def call(self,epoch):
        if epoch < self.warmup_epochs:
            p = self.start_p + self.slope * epoch
        else:
            p = self.max_p = self.max_p
        return p