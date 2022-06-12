import torch
from . import backbones
from . import seg_heads

class Model(torch.nn.Module):
    def __init__(self, backbone_name, backbone_param, seg_head_name, seg_head_param):
        super().__init__()
        self.backbone = getattr(backbones,backbone_name).create_module(**backbone_param)
        self.seg_head = getattr(seg_heads,seg_head_name).create_module(**seg_head_param)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.seg_head(x)
        return {'mask':x}
    
    def predict(self,x):
        with torch.no_grad():
            x = self.backbone(x)
            x = self.seg_head(x)
            x = torch.sigmoid(x)
        return {'mask':x}
    
    def inference(self,output):
        with torch.no_grad():
            mask = output['mask']
            mask = torch.sigmoid(mask)
        return {'mask':mask}