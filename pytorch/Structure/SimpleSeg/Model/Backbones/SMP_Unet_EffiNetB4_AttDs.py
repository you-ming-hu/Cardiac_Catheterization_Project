
import torch
import segmentation_models_pytorch as smp
from utils.model import BaseBackbone

class AttentionDownSample(torch.nn.Module):
    def __init__(self,downscale,in_channel,reduce):
        super().__init__()
        self.downscale = downscale
        self.Q = torch.nn.Linear(in_channel, in_channel//reduce, bias=False)
        self.K = torch.nn.Linear(in_channel, in_channel//reduce, bias=False)
        
    def forward(self,fm):
        B,C,H,W = fm.shape
        new_H = H//self.downscale
        new_W = W//self.downscale
        
        fm = torch.reshape(fm,[B, C, new_H, self.downscale, new_W, self.downscale])
        fm = torch.permute(fm,[0,2,4,3,5,1])
        fm = torch.reshape(fm,[B, new_H, new_W, self.downscale*self.downscale, C])
        
        q = torch.mean(fm,axis=-2,keepdim=True)
        q = self.Q(q)
        q = q / q.shape[-1]**0.5
        
        k = self.K(fm)
        
        qk = torch.matmul(q,torch.transpose(k,-2,-1))
        qk = torch.softmax(qk, dim=-1)
        
        out = torch.matmul(qk,fm)
        out = torch.squeeze(out,dim=-2)
        out = torch.permute(out,[0,3,1,2])
        return out

def stride_conv_modify(layer,reduce):
    layer.downsample = AttentionDownSample(2,layer.out_channels,reduce)
    def forward(self):
        x = torch.nn.functional.conv2d(x, self.weight, self.bias, 1, 'same', self.dilation, self.groups)
        x = self.downsample(x)
        return x
    layer.forward = forward
    
def stride_block_modify(block):
    num_squeezed_channels = max(1, int(block._block_args.input_filters * block._block_args.se_ratio))
    stride_conv_modify(block._depthwise_conv,num_squeezed_channels)

class SMP_Unet_EffiNetB4_AttDs(BaseBackbone):
    def __init__(self, in_channels, out_channels):
        stem = smp.Unet(
            encoder_name = 'efficientnet-b4', 
            encoder_weights = 'imagenet', 
            in_channels = in_channels, 
            classes = out_channels,
            activation = None,
            aux_params = None)
        
        stride_conv_modify(stem.encoder._conv_stem,8)
        
        stride_block_modify(stem.encoder._blocks[2])
        stride_block_modify(stem.encoder._blocks[6])
        stride_block_modify(stem.encoder._blocks[10])
        stride_block_modify(stem.encoder._blocks[22])
        
        non_froozen_layers = [
            stem.encoder._conv_stem,
            stem.encoder._bn0,
            stem.encoder._blocks[2].downsample,
            stem.encoder._blocks[6].downsample,
            stem.encoder._blocks[10].downsample,
            stem.encoder._blocks[22].downsample,
            stem.segmentation_head]
        super().__init__(stem,non_froozen_layers)
        
    def forward(self, x):
        x = self.stem(x)
        return x