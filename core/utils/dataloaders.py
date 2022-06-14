import pathlib
import core.dataset
import core.dataset.readers

class TrainingDataLoader:
    def __init__(self,preprocess,augmentation,postprocess,Config,keep_original=False):
        self.preprocess = preprocess
        self.augmentation = augmentation
        self.postprocess = postprocess
        self.Config = Config
        self.reader = getattr(core.dataset.readers,Config.Data.Type).TrainingDataReader
        self.keep_original = keep_original
        
    def __call__(self,stage,folders):
        Config = self.Config
        stage = Config.AutoGenerate.TrainStages.index(stage)
        preprocess = self.preprocess
        postprocess = self.postprocess
        
        if stage < 2:
            augmentation = self.augmentation
        else:
            augmentation = None
        
        reader = self.reader
        keep_original = self.keep_original
        dataset = core.dataset.Dataset(reader,folders,preprocess,postprocess,augmentation,keep_original)
        
        if stage == 0:
            batch_size=Config.Training.Dataset.Train.BatchSize
            shuffle=True
        else:
            batch_size=Config.Training.Dataset.Val.BatchSize
            shuffle=False
            
        num_workers=Config.Training.Dataset.NumWorkers
        pin_memory=Config.Training.Dataset.PinMemory
        prefetch_factor=Config.Training.Dataset.PrefetchFactor
        
        dataloader = core.dataset.DataLoader(dataset,batch_size,shuffle,num_workers,pin_memory,prefetch_factor)
        return dataloader

class InferenceDataLoader:
    def __init__(self,file_type,root_path,Config):
        assert file_type in ['dcm','png']
        
        self.preprocess = core.dataset.preprocess.PreprocessParser(Config.Data.Image.Preprocess)
        self.postprocess = core.dataset.postprocess.PostprocessParser(Config.Data.Image.Postprocess)
        
        if file_type == 'dcm':
            self.paths = list(pathlib.Path(root_path).rglob('*.dcm'))
            self.reader = getattr(core.dataset.readers,Config.Data.Type).FromDicom
        else:
            self.paths = list(pathlib.Path(root_path).rglob('images'))
            self.reader = getattr(core.dataset.readers,Config.Data.Type).FromImageFolder
            
    def __call__(self,batch_size,num_workers,pin_memory,prefetch_factor):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        return self
    
    def __iter__(self):
        self.i = 0
        return self
    
    def __next__(self):
        if self.i == len(self.paths):
            raise StopIteration
        dataset = core.dataset.Dataset(self.reader,self.paths[self.i],self.preprocess,None,self.postprocess)
        dataset = core.dataset.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor)
        
        self.i += 1
        return self.paths[self.i], dataset