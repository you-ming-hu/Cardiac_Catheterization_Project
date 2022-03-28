
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
            torch.nn.Conv2d(in_channel,reduce,1),
            Lambda(lambda fm : (fm/reduce**0.5).unsqueeze(-1)))
        self.K = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel,reduce,1),
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


def drop_connect(inputs, p, training):
    assert 0 <= p <= 1, 'p must be in range of [0,1]'
    if not training:
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output

def modify_MBConvBlock_ResProp(MBConvBlock):
    in_channels, out_channels = MBConvBlock._block_args.input_filters, MBConvBlock._block_args.output_filters
    num_squeezed_channels = max(1, int(MBConvBlock._block_args.input_filters * MBConvBlock._block_args.se_ratio))
    if MBConvBlock.id_skip and MBConvBlock._block_args.stride == 1 and  in_channels == out_channels:
        MBConvBlock.att_inp = torch.nn.Conv2d(in_channels,num_squeezed_channels,1,bias=False)
        MBConvBlock.att_x = torch.nn.Conv2d(in_channels,num_squeezed_channels,1,bias=False)
        
        def forward(self, inputs, drop_connect_rate=None):
            x = inputs
            if self._block_args.expand_ratio != 1:
                x = self._expand_conv(inputs)
                x = self._bn0(x)
                x = self._swish(x)
            x = self._depthwise_conv(x)
            x = self._bn1(x)
            x = self._swish(x)
            if self.has_se:
                x_squeezed = torch.nn.functional.adaptive_avg_pool2d(x, 1)
                x_squeezed = self._se_reduce(x_squeezed)
                x_squeezed = self._swish(x_squeezed)
                x_squeezed = self._se_expand(x_squeezed)
                x = torch.sigmoid(x_squeezed) * x
            x = self._project_conv(x)
            x = self._bn2(x)
            att = torch.sigmoid(torch.mean(self.att_inp(inputs)*self.att_x(x),dim=1,keepdim=True))
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x*att + inputs*(1-att)
            return x
        
        forward = forward.__get__(MBConvBlock, MBConvBlock.__class__)
        setattr(MBConvBlock, 'forward', forward)
    else:
        pass

class SMP_Unet_EffiNetB4_proirFastAttDs(BaseBackbone):
    def __init__(self, in_channels, out_channels, att_dropout_count, att_res_prop):
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
        
        if att_res_prop:
            for b in stem.encoder._blocks:
                modify_MBConvBlock_ResProp(b)
        
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