import albumentations as A

v1 = A.CLAHE(clip_limit=4,tile_grid_size=(8,8),always_apply=False,p=1)