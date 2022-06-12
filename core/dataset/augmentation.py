import functools
import albumentations as A
from core.lib.schedules import aug_schedules

def AugmentationParser(p,compose):
    p = eval(f'aug_schedules.{p}')
    compose = [eval(f'{a}'.replace('p=','p=aug_schedules.')) for a in compose]
    return Augmentations(p,compose)

class Augmentations:
    def __init__(self,p,compose):
        self.compose = compose
        self.p = p
    def __iter__(self):
        return iter(self.compose)
    def __call__(self,epoch_count):
        assert isinstance(epoch_count,int)
        return A.Sequential([c(epoch_count) for c in self.compose],p=self.p(epoch_count))
    def get_probs(self,epoch_count):
        assert isinstance(epoch_count,int)
        return self.p(epoch_count),{c.name:c.p(epoch_count) for c in self.compose}

class BaseAug:
    def __init__(self,p,func):
        self.p = p
        self.func = func
        self.name = f'{self.__class__.__name__}'
    
    def __call__(self,epoch_count):
        assert isinstance(epoch_count,int)
        return self.func(p=self.p(epoch_count),always_apply=False)

class RandomBrightnessContrast(BaseAug):
    def __init__(self,p,brightness_limit=0.2,contrast_limit=0.2):
        func = functools.partial(
            A.RandomBrightnessContrast,
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            #fixed param
            brightness_by_max=True)
        super().__init__(p,func)

class Downscale(BaseAug):
    def __init__(self,p,scale_min=0.5,scale_max=0.9):
        func =  functools.partial(
            A.Downscale,
            scale_min=scale_min,
            scale_max=scale_max,
            #fixed param
            interpolation=1)
        super().__init__(p,func)
        
class Blur(BaseAug):
    def __init__(self,p,blur_limit=7):
        func = functools.partial(
            A.Blur,
            blur_limit=blur_limit)
        super().__init__(p,func)
        
class GaussNoise(BaseAug):
    def __init__(self,p,var_limit=(0,100),mean=0):
        func = functools.partial(
            A.GaussNoise,
            var_limit=var_limit,
            mean=mean,
            #fixed param
            per_channel=True)
        super().__init__(p,func)
        
class Affine(BaseAug):
    def __init__(self,p,scale=0.9,translate_percent=0.1,rotate=5,shear=5):
        func = functools.partial(
            A.Affine,
            scale=(scale,1/scale),
            translate_percent=(-translate_percent,translate_percent),
            rotate=(-rotate,rotate),
            shear={'x':(-shear,shear),'y':(-shear,shear)},
            #fixed param
            interpolation=1,
            mask_interpolation=0,
            translate_px=None,
            cval=0,
            cval_mask=0,
            mode=0,
            fit_output=False)
        super().__init__(p,func)
        
class ElasticTransform(BaseAug):
    def __init__(self,p,alpha=2000,sigma=60):
        func = functools.partial(
            A.ElasticTransform,
            alpha=alpha,
            sigma=sigma,
            #fixed param
            alpha_affine=0,
            interpolation=1,
            border_mode=0, 
            value=None,
            mask_value=None,
            approximate=False,
            same_dxdy=True)
        super().__init__(p,func)
        
class GridDistortion(BaseAug):
    def __init__(self,p,num_steps=4,distort_limit=0.1):
        func = functools.partial(
            A.GridDistortion,
            num_steps=num_steps,
            distort_limit=distort_limit,
            #fixed param
            interpolation=1,
            border_mode=0,
            value=None,
            mask_value=None)
        super().__init__(p,func)
        
        

        
    
        