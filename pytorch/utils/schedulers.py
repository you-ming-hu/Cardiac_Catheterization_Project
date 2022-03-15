import torch.optim.lr_scheduler as lr_scheduler

class BaseScheduler:
    def __init__(self,scheduler_class,**kwdarg):
        self.scheduler_class = scheduler_class
        self.kwdarg = kwdarg
    def __call__(self,optimizer):
        self.scheduler = self.scheduler_class(optimizer,**self.kwdarg)
        return self
    def state_dict(self):
        return self.scheduler.state_dict()
    def load_state_dict(self,state_dict):
        self.scheduler.load_state_dict(state_dict)
        
class ConstantLR(BaseScheduler):
    def __init__(self,steps_per_epoch,factored_epochs=4,factor=0.5):
        super().__init__(
            scheduler_class=lr_scheduler.ConstantLR,
            factor=factor,
            total_iters=factored_epochs)
    def step(self):
        pass
    def epoch(self,recoreder):
        self.scheduler.step()
    
class LinearLR(BaseScheduler):
    def __init__(self,steps_per_epoch,warmup_epochs,start_factor=0.001,end_factor=1.0):
        super().__init__(
            scheduler_class=lr_scheduler.LinearLR,
            start_factor=start_factor,
            end_factor=end_factor,
            total_iters=steps_per_epoch*warmup_epochs)
    def step(self):
        self.scheduler.step()
    def epoch(self,recoreder):
        pass
        

class StepLR(BaseScheduler):
    def __init__(self,steps_per_epoch,gamma):
        super().__init__(
            scheduler_class = lr_scheduler.StepLR,
            gamma = gamma ** (1/steps_per_epoch),
            step_size = 1
            )
    def step(self):
        self.scheduler.step()
    def epoch(self):
        pass
        
class ReduceLROnPlateau(BaseScheduler):
    def __init__(
        self,
        steps_per_epoch,
        observe_purpose,
        observe_metric,
        mode,
        factor,
        patience,
        threshold=1e-4,
        threshold_mode='rel',
        cooldown=0.,
        min_lr=0.,
        eps=1e-8,
        ):
        super().__init__(
            scheduler_class = lr_scheduler.ReduceLROnPlateau,
            mode = mode,
            factor = factor,
            patience = patience,
            threshold = threshold,
            threshold_mode = threshold_mode,
            cooldown = cooldown,
            min_lr = min_lr,
            eps = eps,
        )
        self.observe_purpose = observe_purpose
        self.observe_metric = observe_metric
    
    def step(self):
        pass
    
    def epoch(self,recorder):
        for m in recorder.metrics[self.observe_purpose]:
            if m.__class__.__name__ == self.observe_metric:
                break
        self.step(m.result())