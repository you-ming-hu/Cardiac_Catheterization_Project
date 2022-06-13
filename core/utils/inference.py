import matplotlib.pyplot as plt
import pathlib

def SegWithClass(save_path,batch_data,output):
    batch_size = batch_data['image'].shape[0]
    for i in range(batch_size):
        mask = batch_data['mask'][i]
        contrast_exist = batch_data['contrast_exist'][i]
        curr_img = batch_data['image'][i][0]
        prev_img = batch_data['image'][i][1]
        
        contrast_exist = batch_data['contrast_exist'][i]
        sample_id = batch_data['sample_id'][i]
        frame_id = batch_data['frame_id'][i]
        
        pred_mask = output['mask'][i]
        pred_contrast_exist =  output['contrast_exist'][i]
        
        save_path = pathlib.Path(save_path,sample_id,frame_id)
        save_path.mkdir(parents=True,exist_ok=True)
        
        plt.imsave(save_path.joinpath('prev'),prev_img,cmap='gray')
        plt.imsave(save_path.joinpath('curr'),curr_img,cmap='gray')
        plt.imsave(save_path.joinpath(f'mask_{contrast_exist:d}'),mask,cmap='gray')
        
        plt.imsave(save_path.joinpath(f'pred_{pred_contrast_exist:.4f}'),pred_mask,cmap='gray')
        
        
        
        