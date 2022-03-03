import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

v1 = A.CLAHE(clip_limit=4,tile_grid_size=(8,8),always_apply=False,p=1)
