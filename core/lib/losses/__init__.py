from . import losses
from core.lib.schedules import loss_schedules

def LossFuncionParser(loss_funcs):
    loss_funcs = [eval(f'losses.{l}'.replace('schedule=','schedule=loss_schedules.')) for l in loss_funcs]
    return LossFunction(loss_funcs)

class LossFunction:
    def __init__(self,loss_funcs):
        self.loss_funcs = loss_funcs
        
    def __iter__(self):
        return iter(self.loss_funcs)
        
    def __call__(self,output,data,step_count=None,steps_per_epoch=None):
        loss_composition = {}
        acc_hybrid_count = 0
        acc_hybrid_loss = 0
        for l in self.loss_funcs:
            acc_count, acc_loss = l(output,data)
            if isinstance(l.schedule,int):
                w = l.schedule
            elif l.schedule is None:
                w = 1
            else:
                w = l.schedule(step_count,steps_per_epoch)
            loss_composition[l.name] = {'acc_count':acc_count.item(),'acc_loss':acc_loss.item(),'weight':w}
            
            acc_hybrid_count += acc_count
            acc_hybrid_loss += acc_loss*w
            
        hybrid_loss = acc_hybrid_loss/acc_hybrid_count if acc_hybrid_count != 0 else None
        return hybrid_loss, loss_composition
    
class LossBuffer:
    def __init__(self,names):
        self.buffers = {n:{'acc_count':0,'acc_loss':0,'acc_weighted_loss':0,'acc_weight':0} for n in names}
    
    def add(self,loss_composition):
        for name, loss in loss_composition.items():
            self.buffers[name]['acc_count'] += loss['acc_count']
            self.buffers[name]['acc_loss'] += loss['acc_loss']
            self.buffers[name]['acc_weighted_loss'] += loss['acc_loss']*loss['weight']
            self.buffers[name]['acc_weight'] += loss['weight']*loss['acc_count']
            
    def result(self):
        loss_composition = {}
        acc_hybrid_count = 0
        acc_hybrid_loss = 0
        for n,l in self.buffers.items():
            if l['acc_count'] != 0:
                loss_composition[n] = {
                    'loss' : l['acc_loss']/l['acc_count'],
                    'weighted_loss' : l['acc_weighted_loss']/l['acc_count'],
                    'weight' : l['acc_weight']/l['acc_count']}
            else:
                loss_composition[n] = {
                    'loss' : None,
                    'weighted_loss' : None,
                    'weight' : None}
            
            acc_hybrid_count += l['acc_count']
            acc_hybrid_loss += l['acc_weighted_loss']
        hybrid_loss = acc_hybrid_loss/acc_hybrid_count if acc_hybrid_count != 0 else None
        return hybrid_loss, loss_composition
    
    def clear(self):
        for n in self.buffers.keys():
            self.buffers[n]['acc_count'] = 0
            self.buffers[n]['acc_loss'] = 0
            self.buffers[n]['acc_weighted_loss'] = 0
            self.buffers[n]['acc_weight'] = 0
            