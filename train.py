import io
import pathlib
import random
import math
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.cuda.amp import autocast, GradScaler


from Dataset import TrainDataLoader

import Transform.Preprocess
import Transform.Combinations

import Schedule.LearningRate

import Tasks

from utils import Recorder
from utils import ModelBuilder

import utils.configuration
Config = utils.configuration.Config

# create gpu device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# cudnn reproducibility
torch.backends.cudnn.deterministic = Config.Training.Settings.Random.cuDNN.Deterministic
torch.backends.cudnn.benchmark = Config.Training.Settings.Random.cuDNN.Benchmark

# create dataset: train_dataset, validation_dataset, validation_dataset_wo_arg
# train_transform for dataloader
train_transform_creater = getattr(Transform.Combinations, Config.Dataset.Train.Transform.Combination.Version)
train_transform_creater = train_transform_creater(
    Config.Dataset.Train.Transform.Schedule,
    Config.Dataset.Train.Transform.Combination.Components,
    Config.Dataset.Train.Transform.Combination.Params,
    Config.Dataset.Train.Transform.Combination.Schedules)
# validation_transform for dataloader
validation_transform_creater = getattr(Transform.Combinations, Config.Dataset.Validation.Transform.Combination.Version)
validation_transform_creater = validation_transform_creater(
    Config.Dataset.Validation.Transform.Schedule,
    Config.Dataset.Validation.Transform.Combination.Components,
    Config.Dataset.Validation.Transform.Combination.Params,
    Config.Dataset.Validation.Transform.Combination.Schedules)
# standard preprocess operation for dataloader
preprocess = getattr(Transform.Preprocess,Config.Dataset.Preprocess.Version)
# create dataloader
dataloader = TrainDataLoader(
    images_root = Config.Dataset.ImagesRootPath,
    masks_root = Config.Dataset.MasksRootPath,
    train_transform = train_transform_creater,
    train_batch_size = Config.Dataset.Train.BatchSize,
    validation_ratio = Config.Dataset.Validation.Ratio,
    validation_transform = validation_transform_creater,
    validation_batch_size = Config.Dataset.Validation.BatchSize,
    image_rgb = Config.Dataset.IO.InputRGBImage,
    preprocess = preprocess,
    num_workers = Config.Dataset.IO.NumWorkers,
    pin_memory = Config.Dataset.IO.PinMemory,
    prefetch_factor = Config.Dataset.IO.PrefetchFactor,
    dtype=Config.Dataset.IO.OutputDtype)
# get datasets from dataloader
# seed for dataset
dataset_transform_seed = Config.Training.Settings.Random.Seed.Dataset.Transform
random.seed(dataset_transform_seed)
dataset_split_seed = Config.Training.Settings.Random.Seed.Dataset.Split
np.random.seed(dataset_split_seed)
train_dataset, validation_dataset, validation_dataset_wo_arg = dataloader.get_dataset()

# calcualte how many steps in an epoch and scheduler need this
train_data_count = len(train_dataset)
gradient_accumulation = Config.Training.Settings.GradientAccumulation
train_batch_size = Config.Dataset.Train.BatchSize
step_size =  gradient_accumulation * train_batch_size
steps_per_epoch = int(math.ceil(train_data_count/step_size))

# Select the task and use correspondent model, loss_fn, metrics
task = getattr(Tasks,Config.Training.Task)
# seed for model creation
model_seed = Config.Training.Settings.Random.Seed.Model
torch.manual_seed(model_seed)
# start building the model
# build the model backbone
model_backbone_class = getattr(task.Model.Backbones,Config.Training.Model.Backbone.Name)
model_backbone_param = Config.Training.Model.Backbone.Param
model_backbone = model_backbone_class(**model_backbone_param)
# build the model head
model_head_class = getattr(task.Model.Heads,Config.Training.Model.Head.Name)
model_head_param = Config.Training.Model.Head.Param
model_head = model_head_class(**model_head_param)
# create the model
model = ModelBuilder(model_backbone,model_head)
model = model.to(device)
# setup loss_fn
loss_fn_class = getattr(task.Losses,Config.Training.Loss.Name)
loss_fn = loss_fn_class(**Config.Training.Loss.Param)
# setup optimizer
optimizer_class = getattr(torch.optim,Config.Training.Optimizer.Name)
optimizer = optimizer_class(model.parameters(),**Config.Training.Optimizer.Param)
# scheduler
scheduler_class = getattr(Schedule.LearningRate,Config.Training.Scheduler.Name)
scheduler = scheduler_class(steps_per_epoch=steps_per_epoch,**Config.Training.Scheduler.Param)
scheduler = scheduler(optimizer)
# setup metrics and setup recoder
metrics_class = [getattr(task.Metrics,m) for m in Config.Training.Metrics.Name]
metrics_params = Config.Training.Metrics.Param
    
# if the model's backbone is not going to be trained, but still keep the input and the output layers trainable. 
if Config.Training.Settings.Model.FreezeBackbone:
    model.freeze_backbone()

# setup for logging validation image segmentation examples
steps_per_log = Config.Logging.StepsPerLog
ncols = Config.Logging.Image.Columns * 3
nrows = Config.Logging.Image.Rows
figsize = Config.Logging.Image.Figsize
fontsize = Config.Logging.Image.Fontsize
dpi = Config.Logging.Image.DPI
mask_alpha = Config.Logging.Image.MaskAlpha
predict_threshold = Config.Logging.Image.Threshold

# create recorder
recorder = Recorder(
    root_path = Config.Logging.RootPath,
    folder_name = Config.Logging.FolderName,
    trial_number=Config.Logging.TrialNumber)
recorder.create_metrics_and_writers(metrics_class,metrics_params)
recorder.log_config(Config)

# amp
amp_scale_train = Config.Training.Settings.AmpScaleTrain
scaler = GradScaler(enabled=amp_scale_train)

# seed for dataset shuffle
dataset_shuffle_seed = Config.Training.Settings.Random.Seed.Dataset.Shuffle
torch.manual_seed(dataset_shuffle_seed)

# start training process
global_step = 0
images_count = 0
for epoch in range(Config.Training.Settings.Epochs):
    acc_count = 0
    acc_loss = 0
    # set model to train mode
    model.train()
    current_datset = train_dataset(epoch)
    for batch_train_data in current_datset:
        images,masks = batch_train_data
        images = images.to(device)
        masks = masks.to(device)
        with autocast(enabled=amp_scale_train,dtype=torch.float32):
            predicts = model(images)
            loss = loss_fn(predicts,masks)['loss']
            recorder.update_metrics_state(purpose='train', predict=predicts, label=masks)

        # update process state
        images_count += images.shape[0]
        acc_loss += loss
        acc_count += 1
    
        # if it's time to do back propagation
        if acc_count == gradient_accumulation:
            acc_loss = acc_loss / gradient_accumulation
            scaler.scale(acc_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            optimizer.zero_grad()
            scheduler.step()
            acc_loss = 0
            acc_count = 0
            global_step += 1
        
            if (global_step % steps_per_log == 0) and global_step != 0:
                recorder.log_metrics_state(purpose='train',images_count=images_count)
                recorder.update_pbar(purpose='train',pbar=current_datset)
                recorder.reset_metrics_state(purpose='train')
                recorder.log_lr(lr=optimizer.param_groups[0]['lr'],images_count=images_count)
    
    
    if global_step % steps_per_log != 0:
        recorder.log_metrics_state(purpose='train',images_count=images_count)
        recorder.reset_metrics_state(purpose='train')
        recorder.log_lr(lr=optimizer.param_groups[0]['lr'],images_count=images_count)
    
    if acc_count != 0:
        acc_loss = acc_loss / acc_count
        scaler.scale(acc_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()
        acc_loss = 0
        acc_count = 0
        global_step += 1
    
    model.eval()
    for purpose, val_dataset in zip(('validation','validation_wo_arg'),(validation_dataset, validation_dataset_wo_arg)):
        
        acc_output_images = 1
        fig = plt.figure(figsize=figsize,dpi=dpi)
        plt.axis(False)
        
        current_datset = val_dataset(epoch)
        for batch_val_data in current_datset:
            images, masks = batch_val_data
            images = images.to(device)
            masks = masks.to(device)
            with autocast(enabled=amp_scale_train,dtype=torch.float32):
                with torch.no_grad():
                    intermediate_predicts = model(images)
                recorder.update_metrics_state(purpose=purpose, predict=intermediate_predicts, label=masks)
            del intermediate_predicts
            recorder.update_pbar(purpose=purpose,pbar=current_datset)
            
            with autocast(enabled=amp_scale_train,dtype=torch.float32):
                final_predicts = model.predict(images)
                
            if acc_output_images <= nrows*ncols:
                for image,mask,predict in zip(images.cpu().numpy().squeeze(axis=1),masks.cpu().numpy(),final_predicts.cpu().detach().numpy()):
                    binary_predict = (predict > predict_threshold).astype(float)
                    tp = (mask==1) & (binary_predict==1)
                    fp = binary_predict > mask
                    fn = mask > binary_predict
                    
                    if acc_output_images < nrows*ncols:
                        fig.add_subplot(nrows,ncols,acc_output_images)
                        plt.imshow(image,cmap='gray')
                        plt.title('Input Image',fontsize=fontsize)
                        plt.axis(False)
                        acc_output_images += 1
                        
                        fig.add_subplot(nrows,ncols,acc_output_images)
                        plt.imshow(image,cmap='gray')
                        plt.imshow(tp,alpha=tp*mask_alpha,cmap='Greens')
                        plt.imshow(fp,alpha=fp*mask_alpha,cmap='Reds')
                        plt.imshow(fn,alpha=fn*mask_alpha,cmap='Blues')
                        plt.title(f'Threshold: {predict_threshold:.2f}',fontsize=fontsize)
                        plt.axis(False)
                        acc_output_images += 1
                        
                        fig.add_subplot(nrows,ncols,acc_output_images)
                        plt.imshow(predict,cmap='gray')
                        plt.title('Predict',fontsize=fontsize)
                        plt.axis(False)
                        acc_output_images += 1
                    
                    del predict
                    del binary_predict
                    del tp
                    del fp
                    del fn
                        
            del final_predicts
            del images
            del masks
        
        scheduler.epoch(recorder)
        
        recorder.log_metrics_state(purpose=purpose,images_count=images_count)
        recorder.reset_metrics_state(purpose=purpose)
        
        #record image
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='raw', dpi=dpi)
        plt.close()
        buf.seek(0)
        img_arr = np.reshape(np.frombuffer(buf.getvalue(), dtype=np.uint8),
                            newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
        buf.close()
        recorder.log_image(purpose=purpose,image=img_arr,epoch=epoch)