import pathlib
import torch
from sklearn.model_selection import train_test_split

from .configuration import  initialize_config, get_config
from .recorder import StageRecorder,LossCompositionRecorder,LearningRateRecorder,AugProbRecorder
from .dataloaders import TrainingDataLoader

import core.dataset.preprocess
import core.dataset.postprocess
import core.dataset.augmentation
import core.dataset

import core.lib.losses
import core.lib.metrics

import core.lib.schedules.lr_schedules

import core.models

parse_format = lambda p: [s.replace('-','(',1).replace(' ','').replace(':','=').replace('-',',')+')' for s in p]

def get_train_dataloaders(Config,keep_original=False):
    train_stages = Config.AutoGenerate.TrainStages
    folders = list(pathlib.Path(Config.Training.Dataset.Path).iterdir())
    train_folders, val_folders = train_test_split(folders, test_size=Config.Training.Dataset.Val.Ratio)
    
    preprocess = core.dataset.preprocess.PreprocessParser(Config.Data.Image.Preprocess)
    postprocess = core.dataset.postprocess.PostprocessParser(Config.Data.Image.Postprocess)
    augmentation = core.dataset.augmentation.AugmentationParser(
        p=Config.Training.Dataset.Augmentation.Prob,
        compose=parse_format(Config.Training.Dataset.Augmentation.Content))
    with Config.AutoGenerate as Auto:
        Auto.AugNames = [a.name for a in augmentation]
    
    DataLoader = TrainingDataLoader(preprocess,augmentation,postprocess,Config,keep_original)
    
    train = DataLoader(train_stages[0],train_folders)
    val = DataLoader(train_stages[1],val_folders)
    val_wo_aug = DataLoader(train_stages[2],val_folders)
    
    dataloaders = {k:v for k,v in zip(train_stages,(train,val,val_wo_aug))}
    
    with Config.AutoGenerate as Auto:
        Auto.StepsPerEpoch = len(dataloaders[train_stages[0]])
    return dataloaders

def get_model(Config):
    task = getattr(core.models,Config.Model.Task)
    model = task.Model(**Config.Model.Param)
    return model

def get_loss_funcs(Config):
    formated_param = parse_format(Config.Training.Losses)
    loss_func = core.lib.losses.LossFuncionParser(formated_param)
    with Config.AutoGenerate as Auto:
        Auto.LossNames = [l.name for l in loss_func]
    return loss_func

def get_metric_funcs(Config):
    formated_param = parse_format(Config.Training.Metrics)
    metric_func = core.lib.metrics.MetricFuncionParser(formated_param)
    with Config.AutoGenerate as Auto:
        Auto.MetricNames = [m.name for m in metric_func]
    return metric_func

def get_loss_buffers(Config):
    return {stage:core.lib.losses.LossBuffer(Config.AutoGenerate.LossNames) for stage in Config.AutoGenerate.TrainStages}
    
def get_metric_buffers(Config):
    return {stage:core.lib.metrics.MetricBuffer(Config.AutoGenerate.MetricNames) for stage in Config.AutoGenerate.TrainStages}

def get_stage_recorders(Config):
    return {stage:StageRecorder(stage,Config.Record.RootPath) for stage in Config.AutoGenerate.TrainStages}
    
def get_loss_composition_recorder(Config):
    return LossCompositionRecorder(Config.AutoGenerate.LossNames,Config.Record.RootPath)

def get_learning_rate_recorder(Config):
    return LearningRateRecorder(Config.Record.RootPath)

def get_aug_prob_recorder(Config):
    return AugProbRecorder(Config.AutoGenerate.AugNames,Config.Record.RootPath)

def get_optimizer(model,Config):
    optimizer = getattr(torch.optim,Config.Training.Optimizer.Name)
    return optimizer(model.parameters(),**Config.Training.Optimizer.Param)

def get_lr_scheduler(optimizer,Config):
    lr_scheduler = getattr(core.lib.schedules.lr_schedules,Config.Training.LearningRateSchedule.Name)
    lr_scheduler = lr_scheduler(steps_per_epoch=Config.AutoGenerate.StepsPerEpoch,**Config.Training.LearningRateSchedule.Param)
    return lr_scheduler(optimizer)

def update_stage_result(dataloader,hybrid_loss,loss_composition,metrics):
    contents = {}
    if hybrid_loss is not None:
        contents['hybrid_loss'] = '{:.4f}'.format(hybrid_loss)
    else:
        contents['hybrid_loss'] = 'None'
    for n,l in loss_composition.items():
        if l['loss'] is not None:
            contents[n] = '{:.4f}'.format(l['loss'])
        else:
            contents[n] = 'None'
    for n,m in metrics.items():
        if m is not None:
            contents[n] = '{:.4f}'.format(m)
        else:
            contents[n] = 'None'
    dataloader.set_postfix(**contents)
    
    