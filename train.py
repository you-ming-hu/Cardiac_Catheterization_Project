import random
import numpy as np
import pathlib
import pickle

import torch
from torch.cuda.amp import autocast, GradScaler

import core.utils

Config = core.utils.get_config()
# create gpu device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# cudnn reproducibility
torch.backends.cudnn.deterministic = Config.Training.Settings.Random.cuDNN.Deterministic
torch.backends.cudnn.benchmark = Config.Training.Settings.Random.cuDNN.Benchmark

random.seed(Config.AutoGenerate.RandomSeed.DatasetAugmentation)
np.random.seed(Config.AutoGenerate.RandomSeed.DatasetSplit)
dataloaders = core.utils.get_train_dataloaders(Config)

metric_func = core.utils.get_metric_funcs(Config)
loss_func = core.utils.get_loss_funcs(Config)

loss_buffers = core.utils.get_loss_buffers(Config)
metric_buffers = core.utils.get_metric_buffers(Config)

torch.manual_seed(Config.AutoGenerate.RandomSeed.ModelWeight)
model = core.utils.get_model(Config)
model = model.to(device)

optimizer = core.utils.get_optimizer(model,Config)
lr_scheduler = core.utils.get_lr_scheduler(optimizer,Config)

stage_recorders = core.utils.get_stage_recorders(Config)
aug_prob_recorder = core.utils.get_aug_prob_recorder(Config)
loss_composition_recorder = core.utils.get_loss_composition_recorder(Config)
learning_rate_recorder = core.utils.get_learning_rate_recorder(Config)

save_config_path = pathlib.Path(Config.Record.RootPath)
save_config_path.mkdir(parents=True)
pickle.dump(Config,save_config_path.joinpath('config.pkl').open('wb'))

stages = Config.AutoGenerate.TrainStages
steps_per_epoch = Config.AutoGenerate.StepsPerEpoch
training_epochs = Config.Training.Settings.Epochs

steps_per_record = Config.Record.Frequence

training_epoch_count = 0
training_step_count = 0
training_data_count = 0

amp_scale_train = Config.Training.Settings.AmpScaleTrain
scaler = GradScaler(enabled=amp_scale_train)

torch.manual_seed(Config.AutoGenerate.RandomSeed.DatasetShuffle)
for _ in range(training_epochs):
    #training
    stage = stages[0]
    print('='*50,f'{stage:0>2}','='*50)
    dataloader = dataloaders[stage](training_epoch_count)
    aug_prob_recorder.write_aug_prob(training_data_count,*dataloaders[stage].get_aug_probs(training_epoch_count))
    
    loss_buffer = loss_buffers[stage]
    metric_buffer = metric_buffers[stage]
    stage_recorder = stage_recorders[stage]
    model.train()
    for batch_data in dataloader:
        batch_data = {k:v.to(device) if not isinstance(v,list) else v for k,v in batch_data.items()}
        with autocast(enabled=amp_scale_train,dtype=torch.float32):
            output = model(batch_data['image'])
            hybrid_loss,loss_composition = loss_func(output,batch_data, step_count = training_step_count, steps_per_epoch = steps_per_epoch)
            metrics = metric_func(output,batch_data)

        if hybrid_loss is not None:
            scaler.scale(hybrid_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            optimizer.zero_grad()
            lr_scheduler.step()
        
        loss_buffer.add(loss_composition)
        metric_buffer.add(metrics)
        
        training_step_count += 1
        training_data_count += batch_data['image'].shape[0]
        
        if training_step_count % steps_per_record == 0:
            hybrid_loss,loss_composition = loss_buffer.result()
            metrics = metric_buffer.result()
            
            stage_recorder.write_loss_result(training_data_count,hybrid_loss,loss_composition)
            stage_recorder.write_metric_result(training_data_count,metrics)
            
            loss_composition_recorder.write_loss_composition(training_data_count,loss_composition)
            learning_rate_recorder.write_learning_rate(training_data_count,optimizer.param_groups[0]['lr'])
            
            core.utils.update_stage_result(dataloader,hybrid_loss,loss_composition,metrics)
            
            loss_buffer.clear()
            metric_buffer.clear()
    
    if training_step_count % steps_per_record != 0:
        hybrid_loss,loss_composition = loss_buffer.result()
        metrics = metric_buffer.result()
        
        stage_recorder.write_loss_result(training_data_count,hybrid_loss,loss_composition)
        stage_recorder.write_metric_result(training_data_count,metrics)
        
        loss_composition_recorder.write_loss_composition(training_data_count,loss_composition)
        learning_rate_recorder.write_learning_rate(training_data_count,optimizer.param_groups[0]['lr'])
    
    loss_buffer.clear()
    metric_buffer.clear()
    
    if Config.Record.SaveModelWeights:
        model_save_path = pathlib.Path(Config.Record.RootPath,'model_weights')
        model_save_path.mkdir(parents=True,exist_ok=True)
        torch.save(model.state_dict(),model_save_path.joinpath(f'{training_epoch_count:0>3}.pt'))
                
    #validation
    model.eval()
    for stage in stages[1:]:
        print('='*50,f'{stage:0>2}','='*50)
        dataloader = dataloaders[stage](training_epoch_count)
        loss_buffer = loss_buffers[stage]
        metric_buffer = metric_buffers[stage]
        stage_recorder = stage_recorders[stage]
        
        for batch_data in dataloader:
            batch_data = {k:v.to(device) if not isinstance(v,list) else v for k,v in batch_data.items()}
            with torch.no_grad():
                with autocast(enabled=amp_scale_train,dtype=torch.float32):
                    output = model(batch_data['image'])
                    hybrid_loss,loss_composition = loss_func(output,batch_data,step_count = training_step_count, steps_per_epoch = steps_per_epoch)
                    metrics = metric_func(output,batch_data)
            
            loss_buffer.add(loss_composition)
            metric_buffer.add(metrics)
            
            hybrid_loss,loss_composition = loss_buffer.result()
            metrics = metric_buffer.result()
            core.utils.update_stage_result(dataloader,hybrid_loss,loss_composition,metrics)
            
            output = model.inference(output)
            core.utils.record_inference(Config,training_epoch_count,stage,batch_data,output)
        
        stage_recorder.write_loss_result(training_data_count,hybrid_loss,loss_composition)
        stage_recorder.write_metric_result(training_data_count,metrics)
        
        loss_buffer.clear()
        metric_buffer.clear()
    
    lr_scheduler.epoch(hybrid_loss)
    training_epoch_count+=1