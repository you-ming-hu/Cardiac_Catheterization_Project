import cv2
import more_itertools
import pydicom

class TrainingDataReader:
    def __init__(self,data_folders,preprocess):
        self.preprocess = preprocess
        self.images = []
        for f in data_folders:
            self.images.extend(more_itertools.pairwise(f.joinpath('images').iterdir()))
            
    def read_image(self,image_path):
        image = cv2.imread(image_path.as_posix())
        return self.preprocess(image)
    
    def read_mask(self,image_path):
        mask_path = image_path.parent.parent.joinpath('masks',image_path.name)
        mask = cv2.imread(mask_path.as_posix(),cv2.IMREAD_GRAYSCALE)
        # assert mask.dtype == np.uint8
        return mask
    
    def __len__(self):
        return len(self.images)
            
    def __getitem__(self,index):
        images = self.images[index]
        prev_image = self.read_image(images[0])
        curr_image = self.read_image(images[1])
        mask = self.read_mask(images[1])
        if len(curr_image.shape) == 2:
            if mask.shape != curr_image.shape:
                mask = cv2.resize(mask,curr_image.shape)
        else:
            if mask.shape != curr_image.shape[1:]:
                mask = cv2.resize(mask,curr_image.shape[1:])
        contrast_exist = 0 if (mask==0).all() else 1
        sample_id = images[1].parent.parent.name
        frame_id = images[1].with_suffix('').name
        return dict(curr_image=curr_image, prev_image=prev_image, mask=mask, contrast_exist=contrast_exist, sample_id=sample_id, frame_id=frame_id)
    
class FromDicom:
    def __init__(self,dicom_path,preprocess):
        self.preprocess = preprocess
        self.dicom_array =  pydicom.read_file(dicom_path.as_posix()).pixel_array
        self.indexes = list(more_itertools.pairwise(range(self.dicom_array.shape[0])))
        
    def __len__(self):
        return len(self.indexes)
        
    def __getitem__(self,index):
        i0, i1 = self.indexes[index]
        prev_image = self.preprocess(self.dicom_array[i0])
        curr_image = self.preprocess(self.dicom_array[i1])
        return dict(curr_image=curr_image, prev_image=prev_image,frame=i1)
    
class FromImageFolder:
    def __init__(self,folder_path,preprocess):
        self.preprocess = preprocess
        self.images = list(more_itertools.pairwise(folder_path.iterdir()))
        
    def read_image(self,image_path):
        image = cv2.imread(image_path.as_posix())
        return self.preprocess(image)
    
    def __getitem__(self,index):
        images = self.images[index]
        prev_image = self.read_image(images[0])
        curr_image = self.read_image(images[1])
        frame = images[1]
        return dict(curr_image=curr_image, prev_image=prev_image,frame=frame)
    