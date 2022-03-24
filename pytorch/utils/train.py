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
    
    def predict(self,x,threshold=None):
        x = self.backbone(x)
        x = self.head.predict(x)
        if threshold != None:
            assert 0 <= threshold <= 1
            x = (x > threshold).type(torch.float32)
        return x
            
class Recorder:
    all_purposes = ['train','validation','validation_wo_arg']
    def __init__(
        self,
        path,
        checkpoint_filename,
        summary_filename):
        save_root = pathlib.Path(path)
        assert not save_root.exists()
        
        self.save_root = save_root
        self.checkpoint_filename = checkpoint_filename
        self.summary_filename = summary_filename
        
    def log_config(self,config):
        text = f'CONFIG:\n  {config}'
        self.writers['train'].add_text('CONFIG',text,0)
        
    def create_metrics_and_writers(self,metrics_class,metrics_param):
        assert len(metrics_class) == len(metrics_param)
        summary_path = self.save_root.joinpath(self.summary_filename)
        self.writers = {p: torch.utils.tensorboard.writer.SummaryWriter(log_dir=summary_path.joinpath(p).as_posix()) for p in self.all_purposes}
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
    
    def save_checkpoint(self,checkpoint):
        checkpoint_path = self.save_root.joinpath(self.checkpoint_filename)
        checkpoint_path.mkdir(parents=True,exist_ok=True)
        checkpoint_count = len(list(checkpoint_path.iterdir()))
        checkpoint_path = checkpoint_path.joinpath(f'checkpoint_{checkpoint_count:0>4}').with_suffix('.tar')
        torch.save(checkpoint,checkpoint_path.as_posix())
        self.checkpoint_path = checkpoint_path
    
    def log_best_checkpoint(self,purpose,epoch,global_step,images_count):
        assert purpose in self.all_purposes
        writer = self.writers['train']
        metrics = self.metrics[purpose]
        for m in metrics:
            is_updated = m.update_best_result()
            if is_updated:
                text = 'epoch: {}, global_step: {}, images_count: {}\n  {}: {}\n  checkpoint_path: {}'.format(
                    epoch,
                    global_step,
                    images_count,
                    str(m),
                    m.best_result,
                    self.checkpoint_path.as_posix())
                writer.add_text(f'best_{m}_on_{purpose}',text,0)
                
    def reset_metrics_state(self,purpose):
        assert purpose in self.all_purposes
        metrics = self.metrics[purpose]
        for m in metrics:
            m.reset_state()
            
    def log_image(self,purpose,image,epoch):
        assert purpose in self.all_purposes
        writer = self.writers['train']
        writer.add_image(f'{purpose} example', image, epoch, dataformats='HWC')
        