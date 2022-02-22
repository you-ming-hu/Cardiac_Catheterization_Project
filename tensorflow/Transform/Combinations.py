import Transform.Schedule
import albumentations as A

class Base:
    def __str__(self):
        return '.'.join([__name__,self.__class__.__name__])+'('+','.join([f'{k}={v}' for k,v in vars(self).items()]) + ')'
    
    def set_default(self,components,schedules):
        assert isinstance(components,(list,tuple))
        assert isinstance(schedules,(list,tuple,int,float,Transform.Schedule.Base))
        self.default_components = components
        self.default_schedules = schedules
        
    def setup(self,transform_schedule,components,params,schedules):
        if isinstance(transform_schedule,(int,float)):
            self.transform_schedule = Transform.Schedule.Constant(transform_schedule)
        else:
            assert isinstance(transform_schedule,Transform.Schedule.Base)
            self.transform_schedule = transform_schedule
        
        if isinstance(components,(list,tuple)):
            self.components = components
        else:
            assert components == 'default'
            self.components = self.default_components
            
        if isinstance(params,list):
            assert len(params) == len(self.components)
            self.params = params
        else:
            assert params == 'default'
            self.params = [{} for _ in self.components]
        
        if isinstance(schedules,(list,tuple,int,float,Transform.Schedule.Base)):
            self.schedules = schedules
        else:
            assert schedules == 'default'
            self.schedules = self.default_schedules
        if isinstance(self.schedules,(int,float)):
            self.schedules = [Transform.Schedule.Constant(self.schedules) for _ in self.components]
        elif isinstance(self.schedules,Transform.Schedule.Base):
            self.schedules = [self.schedules for _ in self.components]
        else:
            self.schedules = [Transform.Schedule.Constant(s) if isinstance(s,(int,float)) else s for s in self.schedules]
        assert len(self.schedules) == len(self.components)
    
    def __call__(self,epoch):
        transforms = [getattr(self,c)(p=s(epoch),**p) for c,p,s in zip(self.components, self.params, self.schedules)]
        return A.Sequential(transforms, p=self.transform_schedule(epoch))
    
class v1(Base):
    def __init__(self,transform_schedule,components='default',params='default',schedules='default'):
        self.set_default(
            components = ['GridDistortion','ElasticTransform','Affine','GaussNoise','Blur','Downscale','RandomBrightnessContrast'],
            schedules = 0.3)
        self.setup(transform_schedule,components,params,schedules)

    def GridDistortion(self,p,num_steps=4,distort_limit=0.1):
        return A.GridDistortion(
            num_steps=num_steps,
            distort_limit=distort_limit,
            p=p,
            #fixed param
            interpolation=1,
            border_mode=0,
            value=None,
            mask_value=None,
            always_apply=False)
    
    def ElasticTransform(self,p,alpha=2000,sigma=60):
        return A.ElasticTransform(
            alpha=alpha,
            sigma=sigma,
            p=p,
            #fixed param
            alpha_affine=0,
            interpolation=1,
            border_mode=0, 
            value=None,
            mask_value=None,
            always_apply=False,
            approximate=False,
            same_dxdy=True)
        
    def Affine(self,p,scale=0.9,translate_percent=0.1,rotate=5,shear=5):
        return A.Affine(
            scale=(scale,1/scale),
            translate_percent=(-translate_percent,translate_percent),
            rotate=(-rotate,rotate),
            shear={'x':(-shear,shear),'y':(-shear,shear)},
            p=p,
            #fixed param
            interpolation=1,
            mask_interpolation=0,
            translate_px=None,
            cval=0,
            cval_mask=0,
            mode=0,
            fit_output=False,
            always_apply=False)
    
    def GaussNoise(self,p,var_limit=(0,100),mean=0):
        return A.GaussNoise(
            var_limit=var_limit,
            mean=mean,
            p=p,
            #fixed param
            per_channel=True,
            always_apply=False)
    
    def Blur(self,p,blur_limit=7):
        return A.Blur(
            blur_limit=blur_limit,
            p=p,
            #fixed param
            always_apply=False)
    
    def Downscale(self,p,scale_min=0.5,scale_max=0.9):
        return A.Downscale(
            scale_min=scale_min,
            scale_max=scale_max,
            p=p,
            #fixed param
            interpolation=1,
            always_apply=False)
    def RandomBrightnessContrast(self,p,brightness_limit=0.2,contrast_limit=0.2):
        return A.RandomBrightnessContrast(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            p=p,
            #fixed param
            brightness_by_max=True,
            always_apply=False)