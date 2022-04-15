import torch.utils.tensorboard
import pathlib
            
class Recorder:
    all_purposes = ['train','validation','validation_wo_arg']
    def __init__(
        self,
        root_path,
        folder_name,
        trial_number):
        
        self.log_path = pathlib.Path(root_path,folder_name,f'{trial_number:0>2}')
        
    def log_config(self,config):
        text = f'CONFIG:  \n{config}'
        self.writers['train'].add_text('CONFIG',text,0)
        
    def create_metrics_and_writers(self,metrics_class,metrics_param):
        assert len(metrics_class) == len(metrics_param)
        self.writers = {p: torch.utils.tensorboard.writer.SummaryWriter(log_dir=self.log_path.joinpath(p).as_posix()) for p in self.all_purposes}
        self.metrics = {p: [mc(**mp) for mc,mp in zip(metrics_class,metrics_param)] for p in self.all_purposes}
        
    def update_metrics_state(self,purpose,predict,label):
        assert purpose in self.all_purposes
        for m in self.metrics[purpose]:
            m.update_state(predict,label)
    
    def log_metrics_state(self,purpose,images_count):
        assert purpose in self.all_purposes
        writer = self.writers[purpose]
        metrics = self.metrics[purpose]
        for m in metrics:
            results = m.result(detail=True)
            for loss_name, loss_value in results.items():
                if loss_name == 'loss':
                    loss_name = str(m)
                writer.add_scalar(loss_name,loss_value,images_count)
                
    def reset_metrics_state(self,purpose):
        assert purpose in self.all_purposes
        metrics = self.metrics[purpose]
        for m in metrics:
            m.reset_state()
    
    def update_pbar(self,purpose,pbar):
        metrics = self.metrics[purpose]
        postfix = {}
        for m in metrics:
            results = m.result(detail=True)
            for loss_name, loss_value in results.items():
                if loss_name == 'loss':
                    loss_name = str(m)
                postfix[loss_name] = f'{loss_value:.6f}'
        pbar.set_postfix(**postfix)
    
    def log_lr(self,lr,images_count):
        writer = self.writers['train']
        writer.add_scalar('learning rate',lr,images_count)
            
    def log_image(self,purpose,image,epoch):
        assert purpose in self.all_purposes
        writer = self.writers['train']
        writer.add_image(f'{purpose} example', image, epoch, dataformats='HWC')
        