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
    def __init__(self,root,reference,derivative,branch,comment,seed,attempt):
        save_root = pathlib.Path(root,reference,derivative,branch,comment,seed)
        if save_root.exists():
            exist_file_count = len(list(save_root.parent.iterdir()))
            new_save_root = save_root.parent.joinpath(f'{seed}_{exist_file_count:0>3}')
            make_sure = input(f'path {save_root.as_posix()} exist, this excution may be duplicated and will use instead the path\n {new_save_root.as_posix()}\n enter y to continue, otherwise raise exception')
            if not make_sure.lower().startswith('y'):
                raise Exception
            save_root = new_save_root
        save_root = save_root.joinpath(f'{attempt:0>4}')
        self.save_root = save_root
        
    def create_metrics_and_writers(self,metrics_class,metrics_param):
        all_purposes = ['train','validation','validation_wo_arg']
        summary_path = self.save_root.joinpath('summary')
        assert len(metrics_class) == len(metrics_param)
        self.writers = {p: torch.utils.tensorboard.writer.SummaryWriter(log_dir=summary_path.joinpath(p).as_posix()) for p in all_purposes}
        self.metrics = {p: [mc(**mp) for mc,mp in zip(metrics_class,metrics_param)] for p in all_purposes}
        
    def log_config(self,config):
        text = f'CONFIG:\n  {config}'
        self.writers['train'].add_text('CONFIG',text,0)
        
    def update_metrics_state(self,purpose,predict,label):
        for m in self.metrics[purpose]:
            m.update_state(predict,label)
    
    def log_metrics(self,purpose,global_step):
        writer = self.writers[purpose]
        metrics = self.metrics[purpose]
        for m in metrics:
            result = m.result()
            writer.add_scalar(str(m),result,global_step)
            if purpose != 'train':
                is_updated = m.update_best_result()
                if is_updated:
                    text = '\n  '.join(['Epoch',''])
                    writer.add_text(f'best_{m}',text,global_step)
                # logbest
            m.reset_state()
        
        
Epoch: 
Step: 20959
Accuracy: 0.9936708807945251
Weight: logs/model_1/trial_2/weights/0013/weights