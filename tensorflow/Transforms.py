import warnings
warnings.filterwarnings('error',category=SyntaxWarning)

import albumentations as A

def combination_v1(
       Transfrom_p=0.5,
       
       GridDistortion=True,
       GridDistortion_num_steps=4,
       GridDistortiondistort_limit=0.1,
       GridDistortion_p=0.3,
       
       ElasticTransform = True,
       ElasticTransform_alpha=2000,
       ElasticTransform_sigma=60,
       ElasticTransform__p=0.3,
       
       Affine=True,
       Affine_scale=0.9,
       Affine_translate_percent=0.1,
       Affine_rotate=5,
       Affine_shear=5,
       Affine_p=0.3,
       
       GaussNoise = True,
       GaussNoise_var_limit=(0,100),
       GaussNoise_mean=0,
       GaussNoise_p=0.3,
       
       Blur = True,
       Blur_blur_limit=7,
       Blur_p=0.3,
       
       Downscale = True,
       Downscale_scale_min=0.5,
       Downscale_scale_max=0.9,
       Downscale_p=0.3,
       
       RandomBrightnessContrast = True,
       RandomBrightnessContrast_brightness_limit=0.2,
       RandomBrightnessContrast_contrast_limit=0.2,
       RandomBrightnessContrast_p=0.3
       ):
       
       comb = []
       
       #格點失真
       if GridDistortion is True:
              comb.append(
                     A.GridDistortion(
                            num_steps=GridDistortion_num_steps,
                            distort_limit=GridDistortiondistort_limit,
                            p=GridDistortion_p,
                            #fixed param
                            interpolation=1,
                            border_mode=0,
                            value=None,
                            mask_value=None,
                            always_apply=False)
                     )
       
       #非剛性變換
       if ElasticTransform is True:
              comb.append(
                     A.ElasticTransform(
                            alpha=ElasticTransform_alpha,
                            sigma=ElasticTransform_sigma,
                            p=ElasticTransform__p,
                            #fixed param
                            alpha_affine=0,
                            interpolation=1,
                            border_mode=0, 
                            value=None,
                            mask_value=None,
                            always_apply=False,
                            approximate=False,
                            same_dxdy=True)
                     )
    
       #縮放,平移,旋轉,切變
       if Affine is True:
              comb.append(
                     A.Affine(
                            scale=(Affine_scale,1/Affine_scale),
                            translate_percent=(-Affine_translate_percent,Affine_translate_percent),
                            rotate=(-Affine_rotate,Affine_rotate),
                            shear={'x':(-Affine_shear,Affine_shear),'y':(-Affine_shear,Affine_shear)},
                            p=Affine_p,
                            #fixed param
                            interpolation=1,
                            mask_interpolation=0,
                            translate_px=None,
                            cval=0,
                            cval_mask=0,
                            mode=0,
                            fit_output=False,
                            always_apply=False)
                     )
    
       #噪音       
       if GaussNoise is True:
              comb.append(
                     A.GaussNoise(
                            var_limit=GaussNoise_var_limit,
                            mean=GaussNoise_mean,
                            p=GaussNoise_p,
                            #fixed param
                            per_channel=True,
                            always_apply=False)
                     )
    
       #模糊
       if Blur is True:
              comb.append(
                     A.Blur(
                            blur_limit=Blur_blur_limit,
                            p=Blur_p,
                            #fixed param
                            always_apply=False)
                     )
    
       #下採樣模糊
       if Downscale is True:
              comb.append(
                     A.Downscale(
                            scale_min=Downscale_scale_min,
                            scale_max=Downscale_scale_max,
                            p=Downscale_p,
                            #fixed param
                            interpolation=1,
                            always_apply=False)
                     )
    
       #亮度,對比度
       if RandomBrightnessContrast is True:
              comb.append(
                     A.RandomBrightnessContrast(
                            brightness_limit=RandomBrightnessContrast_brightness_limit,
                            contrast_limit=RandomBrightnessContrast_contrast_limit,
                            p=RandomBrightnessContrast_p,
                            #fixed param
                            brightness_by_max=True,
                            always_apply=False)
                     )
    
       return A.Compose(comb,p=Transfrom_p)
    