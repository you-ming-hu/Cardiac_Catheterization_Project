import torch
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import pathlib
from sklearn.model_selection import train_test_split

class Dataset(torch.utils.data.Dataset):
    def __init__(self,data_pairs,image_rgb,transform,dtype):
        self.data_pairs = data_pairs
        self.image_rgb = image_rgb
        if dtype == 'float' or dtype == float:
            self.transform = A.Compose([
                transform,
                A.ToFloat(p=1),
                ToTensorV2(p=1)],p=1)
        else:
            assert dtype == 'int' or dtype == int
            self.transform = A.Compose([
                transform,
                ToTensorV2(p=1)],p=1)
        
    def __getitem__(self,index):
        image_path, mask_path = self.data_pairs[index]
        if isinstance(image_path,pathlib.Path):
            image_path = image_path.as_posix()
        if isinstance(mask_path,pathlib.Path):
            mask_path = mask_path.as_posix()
        
        if self.image_rgb:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        
        mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
        sample = self.transform(image=image,mask=mask)
        image = sample['image']
        mask = sample['mask']
        return image, mask #, image_path
    
    def __len__(self):
        return len(self.data_pairs)
    
class DataLoader:
    def __init__(
        self,
        images_path,
        masks_path,
        image_rgb,
        transform,
        preprocess,
        batch_size,
        drop_last,
        shuffle,
        num_workers,
        pin_memory,
        prefetch_factor,
        dtype):
        
        self.data_pairs = list(zip(images_path,masks_path))
        self.image_rgb = image_rgb
        if transform is None:
            transform = A.NoOp(p=1)
        self.transform = transform
        if preprocess is None:
            preprocess = A.NoOp(p=1)
        self.preprocess = preprocess
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.dtype = dtype
    
    def __len__(self):
        return len(self.data_pairs)
        
    def __call__(self,epoch):
        transform = A.Sequential([self.transform(epoch),self.preprocess],p=1)
        dataset = Dataset(self.data_pairs,self.image_rgb,transform,self.dtype)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor)
        return dataloader
        
class TrainDataLoader:
    def __init__(
        self,
        images_root,
        masks_root,
        train_transform,
        train_batch_size,
        validation_ratio,
        validation_transform,
        validation_batch_size,
        image_rgb,
        preprocess,
        num_workers,
        pin_memory,
        prefetch_factor,
        dtype):
        
        images_path = list(pathlib.Path(images_root).rglob('*.png'))
        masks_path = list(pathlib.Path(masks_root).rglob('*.png'))
        
        self.images_path = images_path
        self.masks_path = masks_path
        
        self.train_transform = train_transform
        self.train_batch_size = train_batch_size
        
        self.validation_ratio = validation_ratio
        self.validation_transform = validation_transform
        self.validation_batch_size = validation_batch_size
        
        self.image_rgb = image_rgb
        self.preprocess = preprocess
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.dtype = dtype
        
        self.is_match = False
        assert len(images_path) == len(masks_path), f'the number images is : {len(images_path)}, while the number of masks is {len(masks_path)}, and they do not match'
        for i, (image_path, mask_path) in enumerate(zip(images_path,masks_path)):
            assert image_path.name == mask_path.name, f'pair {i}, image: {image_path.name} and mask: {mask_path.name} do not match'
        self.is_match = True
    
    def get_dataset(self):
        if not self.is_match:
            raise Exception('images and masks do not match')
        
        train_images_path, validation_images_path, train_masks_path, validation_masks_path = train_test_split(
            self.images_path,
            self.masks_path,
            test_size = self.validation_ratio)
        
        train_dataset = DataLoader(
            images_path = train_images_path,
            masks_path = train_masks_path,
            image_rgb = self.image_rgb,
            transform = self.train_transform,
            preprocess = self.preprocess,
            batch_size = self.train_batch_size,
            drop_last = False,
            shuffle = False,
            num_workers = self.num_workers,
            pin_memory = self.pin_memory,
            prefetch_factor = self.prefetch_factor,
            dtype = self.dtype)
        
        validation_dataset = DataLoader(
            images_path = validation_images_path,
            masks_path = validation_masks_path,
            image_rgb = self.image_rgb,
            transform = self.validation_transform,
            preprocess = self.preprocess,
            batch_size = self.validation_batch_size,
            drop_last = False,
            shuffle = False,
            num_workers = self.num_workers,
            pin_memory = self.pin_memory,
            prefetch_factor = self.prefetch_factor,
            dtype = self.dtype)
        
        validation_dataset_wo_arg = DataLoader(
            images_path = validation_images_path,
            masks_path = validation_masks_path,
            image_rgb = self.image_rgb,
            transform = None,
            preprocess = self.preprocess,
            batch_size = self.validation_batch_size,
            drop_last = False,
            shuffle = False,
            num_workers = self.num_workers,
            pin_memory = self.pin_memory,
            prefetch_factor = self.prefetch_factor,
            dtype = self.dtype)
        
        return train_dataset, validation_dataset , validation_dataset_wo_arg
    
# class Images:
#     def __init__(self, images_path, transform, preprocess, ):
#         if transform is None:
#             transform = A.NoOp()
        
#         self.transform = A.Compose([transform,preprocess],p=1)
#         pass