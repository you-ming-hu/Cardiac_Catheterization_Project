import torch
import albumentations as A
from tqdm import tqdm

class DataProcessor:
    def __init__(self,postprocess,augmentation=None,keep_original=False):
        self.postprocess = postprocess
        self.augmentation = augmentation
        self.keep_original = keep_original
        self.set_epoch(0)
        
    def __call__(self,curr_image,prev_image=None,mask=None,contrast_exist=None,**kwdarg):
        inputs = {'image':curr_image}
        if prev_image is not None:
            inputs['prev_image'] = prev_image
        if mask is not None:
            inputs['mask'] = mask
        
        data = self.transform(self.process,inputs)
        if self.keep_original:
            data['original'] = self.transform(self.process_wo_aug,inputs)
            
        if contrast_exist is not None:
            data['contrast_exist'] = float(contrast_exist)
        data.update(kwdarg)
        return data
    
    def transform(self,process,inputs):
        sample = process(**inputs)
        data = {}
        if 'prev_image' in inputs.keys():
            curr_image = sample['image']
            prev_image = sample['prev_image']
            data['image'] = torch.concat([curr_image,prev_image,curr_image-prev_image],dim=0)
        else:
            data['image'] = sample['image']
        if 'mask' in inputs.keys():
            data['mask'] = sample['mask'].type(torch.float32)/255
        return data
        
    def set_epoch(self,epoch_count):
        additional_targets = {'prev_image': 'image'}
        if self.augmentation is None:
            self.process = A.Compose([self.postprocess],p=1,additional_targets=additional_targets)
        elif isinstance(self.augmentation,(A.core.transforms_interface.BasicTransform,A.core.composition.BaseCompose)):
            self.process = A.Compose([self.augmentation,self.postprocess],p=1,additional_targets=additional_targets)
        else:
            self.process = A.Compose([self.augmentation(epoch_count),self.postprocess],p=1,additional_targets=additional_targets)
            
        if self.keep_original:
            self.process_wo_aug = A.Compose([self.postprocess],p=1,additional_targets=additional_targets)
            
            
class Dataset(torch.utils.data.Dataset):
    def __init__(self,reader,inputs,preprocess,postprocess,augmentation=None,keep_original=False):
        super().__init__()
        self.dataset = reader(inputs,preprocess)
        self.processor = DataProcessor(postprocess,augmentation,keep_original)
        
    def __getitem__(self,index):
        return self.processor(**self.dataset[index])
    
    def __len__(self):
        return len(self.dataset)
    
    def __call__(self,epoch_count):
        self.processor.set_epoch(epoch_count)
        return self
        
class DataLoader:
    def __init__(self,dataset,batch_size,shuffle,num_workers,pin_memory,prefetch_factor):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
    
    def __len__(self):
        return len(torch.utils.data.DataLoader(self.dataset,batch_size=self.batch_size,drop_last=False))
    
    def __call__(self,epoch_count=None):
        if epoch_count is not None:
            dataset = self.dataset(epoch_count)
        else:
            dataset = self.dataset
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor)
        return tqdm(dataloader, position = 0, leave = True, desc=f'EPOCH: {epoch_count}')
    
    def get_aug_probs(self,epoch_count):
        return self.dataset.processor.augmentation.get_probs(epoch_count)