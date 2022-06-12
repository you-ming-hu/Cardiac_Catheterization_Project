class BaseSchedule:
    def __call__(self,step_count,steps_per_epoch):
        if step_count is None and steps_per_epoch is None:
            return 1
        else:
            assert not step_count is None and not steps_per_epoch is None
            return self.call(step_count,steps_per_epoch)
    def call(self,step_count,steps_per_epoch):
        raise NotImplementedError

class Constant:
    def __init__(self,v):
        self.v = v
    def call(self,step_count,steps_per_epoch):
        return self.v
    
class WarmUp:
    def __init__(self,warmup_epochs,max_weight):
        pass
    
    
class Zip:
    def __init__(self,):
        pass
    
class LinearDecay:
    def __init__(self):
        pass