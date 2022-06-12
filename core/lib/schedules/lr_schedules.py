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
    def epoch(self,hybrid_loss):
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
    def epoch(self,hybrid_loss):
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
    def epoch(self,hybrid_loss):
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
    
    def epoch(self,hybrid_loss):
        if hybrid_loss is not None:
            self.scheduler.step(hybrid_loss)
        
        
class Custom_1:
    def __init__(self, steps_per_epoch, warmup_epochs, reduce_gamma=-0.5):
        super().__init__()
        self.warmup_steps = steps_per_epoch * warmup_epochs
        assert reduce_gamma < 0
        self.reduce_gamma = reduce_gamma
        self.step_count = 1
    
    def __call__(self,optimizer):
        assert len(optimizer.param_groups) == 1
        self.optimizer = optimizer
        self.init_lr = optimizer.param_groups[0]['lr']
        return self
        
    def step(self):
        arg1 = self.step_count ** self.reduce_gamma
        arg2 = self.step_count * (self.warmup_steps ** (self.reduce_gamma-1))
        
        self.optimizer.param_groups[0]['lr'] = self.init_lr * (self.warmup_steps**-self.reduce_gamma) * min(arg1, arg2)
        self.step_count += 1

    def epoch(self,hybrid_loss):
        pass
        
    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)