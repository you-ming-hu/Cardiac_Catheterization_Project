import pathlib
        
from torch.utils.tensorboard.writer import SummaryWriter

class StageRecorder:
    def __init__(self,stage,rootpath):
        rootpath = pathlib.Path(rootpath,'record')
        self.loss_writer = SummaryWriter(rootpath.joinpath('loss',stage).as_posix())
        self.metric_writer = SummaryWriter(rootpath.joinpath('metric',stage).as_posix())
    def write_loss_result(self,training_data_count,hybrid_loss,loss_composition):
        for n,l in loss_composition.items():
            if l['loss'] is not None:
                self.loss_writer.add_scalar(n,l['loss'],training_data_count)
        if hybrid_loss is not None:
            self.loss_writer.add_scalar('hybrid_loss',hybrid_loss,training_data_count)
    def write_metric_result(self,training_data_count,metric_result):
        for n,m in metric_result.items():
            if m is not None:
                self.metric_writer.add_scalar(n,m,training_data_count)
    
class LossCompositionRecorder:
    def __init__(self,names,rootpath):
        rootpath = pathlib.Path(rootpath,'record')
        self.writers = {n:SummaryWriter(rootpath.joinpath('loss_composition',n).as_posix()) for n in names}
    def write_loss_composition(self,training_data_count,loss_composition):
        for n,l in loss_composition.items():
            if l['weight'] is not None:
                self.writers[n].add_scalar('weight',l['weight'],training_data_count)
                self.writers[n].add_scalar('weighted_loss',l['weighted_loss'],training_data_count)

class LearningRateRecorder:
    def __init__(self,rootpath):
        rootpath = pathlib.Path(rootpath,'record')
        self.writer = SummaryWriter(rootpath.joinpath('learning_rate').as_posix())
    def write_learning_rate(self,training_data_count,lr):
        self.writer.add_scalar('learning_rate',lr,training_data_count)
        
class AugProbRecorder:
    def __init__(self,names,rootpath):
        rootpath = pathlib.Path(rootpath,'record')
        self.writers = {n:SummaryWriter(rootpath.joinpath('augmentation_prob',n).as_posix()) for n in names}
        self.aug_prob_writer = SummaryWriter(rootpath.joinpath('augmentation_prob','Aug_prob').as_posix())
    def write_aug_prob(self,training_data_count,aug_prob,probs):
        for n,p in probs.items():
            self.writers[n].add_scalar('augmentation_prob',p,training_data_count)
        self.aug_prob_writer.add_scalar('augmentation_prob',aug_prob,training_data_count)
            
        