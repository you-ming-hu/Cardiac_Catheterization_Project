import torch
import torch.utils.tensorboard
import pathlib

class ModelBuilder(torch.nn.Module):
    def __init__(self,backbone,head):
        super().__init__()
        self.backbone = backbone
        self.head = head
        
    def freeze_backbone(self):
        self.backbone.freeze()
        
    def forward(self,x):
        x = self.backbone(x)
        x = self.head(x)
        return x
            
class Recorder:
    def __init__(self,root,reference,derivative,branch,comment,purpose,dataset_seed,model_seed):
        save_root = pathlib.Path(root,reference,derivative,branch,comment,f'seed(d{dataset_seed},m{model_seed})')
        if save_root.exists():
            exist_file_count = len(list(save_root.parent.iterdir()))
            new_save_root = save_root.parent.joinpath(f'seed(d{dataset_seed},m{model_seed})_{exist_file_count:0>3}')
            make_sure = input(f'path {save_root.as_posix()} exist, this excution may be duplicated and will use instead the path\n {new_save_root.as_posix()}\n enter y to continue, otherwise raise exception')
            if not make_sure.lower().startswith('y'):
                raise Exception
            save_root = new_save_root
        save_root = save_root.joinpath(purpose)
        self.save_root = save_root
        
    def log_config(self,config):
        text = f'CONFIG:\n  {config}'
        self.writers['train'].add_text('CONFIG',text,0)
        
    def create_metrics_and_writers(self,metrics_class,metrics_param):
        all_purposes = ['train','validation','validation_wo_arg']
        summary_path = self.save_root.joinpath('summary')
        assert len(metrics_class) == len(metrics_param)
        self.writers = {p: torch.utils.tensorboard.writer.SummaryWriter(log_dir=summary_path.joinpath(p).as_posix()) for p in all_purposes}
        self.metrics = {p: [mc(**mp) for mc,mp in zip(metrics_class,metrics_param)] for p in all_purposes}
        
    def update_metrics_state(self,purpose,predict,label):
        for m in self.metrics[purpose]:
            m.update_state(predict,label)
    
    def log_metrics_state(self,purpose,images_count):
        writer = self.writers[purpose]
        metrics = self.metrics[purpose]
        for m in metrics:
            result = m.result()
            writer.add_scalar(str(m),result,images_count)
    
    def log_lr(self,lr,images_count):
        writer = self.writers['train']
        writer.add_scalar('learning rate',lr,images_count)
    
    def save_checkpoint(self,purpose,checkpoint):
        writer = self.writers['train']
        metrics = self.metrics[purpose]
        checkpoint_path = self.save_root.joinpath('checkpoint')
        checkpoint_count = len(list(checkpoint_path.iterdir()))
        checkpoint_path.joinpath(f'checkpoint_{checkpoint_count:0>4}').with_suffix('.tar')
        torch.save(checkpoint,checkpoint_path.as_posix())
        
        for m in metrics:
            is_updated = m.update_best_result()
            if is_updated:
                text = 'epoch: {}, global_step: {}, images_count: {}\n  {}: {}\n  checkpoint_path: {}'.format(
                    checkpoint['epoch'],
                    checkpoint['global_step'],
                    checkpoint['images_count'],
                    str(m),
                    m.best_result,
                    checkpoint_path.as_posix())
                writer.add_text(f'best_{m}_on_{purpose}',text,0)
                
    def reset_metrics_state(self,purpose):
        metrics = self.metrics[purpose]
        for m in metrics:
            m.reset_state()
            
    def log_image(self,purpose,image,epoch):
        writer = self.writers['train']
        writer.add_image(f'{purpose} example', image, epoch, dataformats='HWC')
        