import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def PostprocessParser(func_str):
    return eval(func_str)

CLAHE_float = A.Compose([
    A.CLAHE(clip_limit=(4,4),tile_grid_size=(8,8),always_apply=False,p=1),
    A.ToFloat(p=1),
    ToTensorV2(p=1)],p=1)
    
CLAHE_int = A.Compose([
    A.CLAHE(clip_limit=(4,4),tile_grid_size=(8,8),always_apply=False,p=1),
    ToTensorV2(p=1)],p=1)