
import torch
import segmentation_models_pytorch as smp
from utils.model import BaseBackbone

class Lambda(torch.nn.Module):
    def __init__(self,fn):
        super().__init__()
        self.fn = fn
    def forward(self,x):
        return self.fn(x)
    
class ConverToGrid(torch.nn.Module):
    def __init__(self,S):
        super().__init__()
        self.S = S
        
    def forward(self,fm):
        S = self.S
        B,C,H,W = fm.shape
        H = H//S
        W = W//S
        fm = torch.reshape(fm,[B, C, H, S, W, S])
        fm = torch.permute(fm,[0,1,2,4,3,5])
        fm = torch.reshape(fm,[B, C, H, W, S*S])
        return fm

class AttentionDownSample(torch.nn.Module):
    def __init__(self,downscale,in_channel,reduce,dropout_count):
        super().__init__()
        self.Q = torch.nn.Sequential(
            torch.nn.AvgPool2d(downscale,downscale),
            torch.nn.Conv2d(in_channel,reduce,1,bias=False),
            Lambda(lambda fm : (fm/reduce**0.5).unsqueeze(-1)))
        self.K = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel,reduce,1,bias=False),
            ConverToGrid(downscale))
        self.V = ConverToGrid(downscale)
        self.dropout = torch.nn.Dropout(p=dropout_count/downscale**2,inplace=True)
        
    def forward(self,fm):
        Q = self.Q(fm)
        K = self.K(fm)
        fm = self.V(fm)
        fm = fm * self.dropout(torch.nn.functional.softmax(torch.sum(Q*K,dim=1,keepdim=True),dim=-1))
        fm = torch.sum(fm,dim=-1) 
        return fm

def stride_conv_modify(layer,reduce,att_dropout_count):
    layer.downsample = AttentionDownSample(2,layer.in_channels,reduce,att_dropout_count)
    layer.padding = 'same'
    layer.stride = (1,1)
    del layer.static_padding
    def forward(self,x):
        x = self.downsample(x)
        x = torch.nn.functional.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x
    forward = forward.__get__(layer, layer.__class__)
    setattr(layer, 'forward', forward)
    
def stride_block_modify(block,att_dropout):
    num_squeezed_channels = max(1, int(block._block_args.input_filters * block._block_args.se_ratio))
    stride_conv_modify(block._depthwise_conv,num_squeezed_channels,att_dropout)


class Unet_EffNet_AttDs(BaseBackbone):
    def __init__(self, in_channels, out_channels, att_dropout_count):
        stem = smp.Unet(
            encoder_name = 'efficientnet-b4', 
            encoder_weights = 'imagenet', 
            in_channels = in_channels, 
            classes = out_channels,
            activation = None,
            aux_params = None)
        
        stride_conv_modify(stem.encoder._conv_stem,8,att_dropout_count*0)
        
        stride_block_modify(stem.encoder._blocks[2],att_dropout_count*1/5)
        stride_block_modify(stem.encoder._blocks[6],att_dropout_count*2/5)
        stride_block_modify(stem.encoder._blocks[10],att_dropout_count*3/5)
        stride_block_modify(stem.encoder._blocks[22],att_dropout_count*4/5)
        
        non_froozen_layers = [
            stem.encoder._conv_stem,
            stem.encoder._bn0,
            stem.encoder._blocks[2]._depthwise_conv.downsample,
            stem.encoder._blocks[6]._depthwise_conv.downsample,
            stem.encoder._blocks[10]._depthwise_conv.downsample,
            stem.encoder._blocks[22]._depthwise_conv.downsample,
            stem.segmentation_head]
        super().__init__(stem,non_froozen_layers)
        
    def forward(self, x):
        x = self.stem(x)
        return x