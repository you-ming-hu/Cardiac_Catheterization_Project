import torch

from . import backbones
from . import seg_heads
from . import classify_heads

class Model(torch.nn.Module):
    def __init__(self, backbone_name, backbone_param, seg_head_name, seg_head_param,classify_heads_name,classify_heads_param):
        super().__init__()
        self.backbone = getattr(backbones,backbone_name).create_module(**backbone_param)
        self.seg_head = getattr(seg_heads,seg_head_name).create_module(**seg_head_param)
        self.classify_head = getattr(classify_heads,classify_heads_name).create_module(**classify_heads_param)
        
    def forward(self, x):
        output = self.backbone(x)
        mask = self.seg_head(output['seg'])
        contrast_exist = self.classify_head(output['classify'])
        return {'mask':mask,'contrast_exist':contrast_exist}
    
    def predict(self,x):
        with torch.no_grad():
            output = self(x)
        return self.inference(output)

    def inference(self,output):
        with torch.no_grad():
            mask = output['mask']
            mask = torch.sigmoid(mask)
            
            contrast_exist = output['contrast_exist']
            contrast_exist = torch.sigmoid(contrast_exist)
        mask = mask.cpu().numpy()
        contrast_exist = contrast_exist.cpu().numpy()
        return {'mask':mask,'contrast_exist':contrast_exist}