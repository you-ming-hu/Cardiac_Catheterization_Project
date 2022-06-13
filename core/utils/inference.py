import matplotlib.pyplot as plt
import pathlib

def SegWithClass(save_path,batch_data,output):
    batch_size = batch_data['image'].shape[0]
    batch_data['mask'] = batch_data['mask'].cpu().numpy()
    batch_data['contrast_exist'] = batch_data['contrast_exist'].cpu().numpy()
    batch_data['image'] = batch_data['image'].cpu().numpy()
    
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
        
        plt.imsave(save_path.joinpath('prev.png'),prev_img,cmap='gray')
        plt.imsave(save_path.joinpath('curr.png'),curr_img,cmap='gray')
        plt.imsave(save_path.joinpath(f'mask_{int(contrast_exist):d}.png'),mask,cmap='gray')
        
        plt.imsave(save_path.joinpath(f'pred_{pred_contrast_exist:.4f}.png'),pred_mask,cmap='gray')
        
        
        
        