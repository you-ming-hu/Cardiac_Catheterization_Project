import torch
import segmentation_models_pytorch as smp
from core.model.base.backbone import BaseBackbone

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
            x = x*2
            return x
        
        forward = forward.__get__(MBConvBlock, MBConvBlock.__class__)
        setattr(MBConvBlock, 'forward', forward)
    else:
        pass


def create_module(in_channels, out_channels):
    module =  smp.Unet(
        encoder_name = 'efficientnet-b4', 
        encoder_weights = 'imagenet', 
        in_channels = in_channels, 
        classes = out_channels,
        activation = None,
        aux_params = None)
    
    for b in module.encoder._blocks:
        modify_MBConvBlock_ResProp(b)
        
    return module