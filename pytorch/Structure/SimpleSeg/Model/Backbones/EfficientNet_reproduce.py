import torch
import math

efficientnet_b4_parameters = [
    dict(num_repeat=2, kernel_size=3, stride=1, expand_ratio=1, input_filters=48, output_filters=24, se_reduce=4),
    dict(num_repeat=4, kernel_size=3, stride=2, expand_ratio=6, input_filters=24, output_filters=32, se_reduce=4),
    dict(num_repeat=4, kernel_size=5, stride=2, expand_ratio=6, input_filters=32, output_filters=56, se_reduce=4),
    dict(num_repeat=6, kernel_size=3, stride=2, expand_ratio=6, input_filters=56, output_filters=112, se_reduce=4),
    dict(num_repeat=6, kernel_size=5, stride=1, expand_ratio=6, input_filters=112, output_filters=160, se_reduce=4),
    dict(num_repeat=8, kernel_size=5, stride=2, expand_ratio=6, input_filters=160, output_filters=272, se_reduce=4),
    dict(num_repeat=2, kernel_size=3, stride=1, expand_ratio=6, input_filters=272, output_filters=448, se_reduce=4)]

def drop_connect(inputs, p, training):
    assert 0 <= p <= 1, 'p must be in range of [0,1]'

    if not training:
        return inputs

    batch_size = inputs.shape[0]
    keep_prob = 1 - p

    # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)

    output = inputs / keep_prob * binary_tensor
    return output


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(torch.nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)
    
class Conv2dDynamicSamePadding(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = [self.stride]*2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)  # change the output size according to stride ! ! !
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return torch.nn.functional.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class MBConvBlock(torch.nn.Module):
    def __init__(self, kernel_size, stride, expand_ratio, input_filters, output_filters, se_reduce):
        super().__init__()
        bn_mom = 1 - 0.99
        bn_eps = 0.001
        
        self.id_skip = (stride == 1) and (input_filters == output_filters)
        
        inp = input_filters
        oup = input_filters * expand_ratio
        if expand_ratio != 1:
            self._expand = True
            self._expand_conv = torch.nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = torch.nn.BatchNorm2d(num_features=oup, momentum=bn_mom, eps=bn_eps)
        else:
            self._expand = False
        
        self._depthwise_conv = Conv2dDynamicSamePadding(in_channels=oup, out_channels=oup, groups=oup, kernel_size=kernel_size, stride=stride, bias=False, padding='same')
        self._bn1 = torch.nn.BatchNorm2d(num_features=oup, momentum=bn_mom, eps=bn_eps)
        
        num_squeezed_channels = max(1, int(input_filters//se_reduce))
        self._se_reduce = torch.nn.Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
        self._se_expand = torch.nn.Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)
        
        self._project_conv = torch.nn.Conv2d(in_channels=oup, out_channels=output_filters, kernel_size=1, bias=False)
        self._bn2 = torch.nn.BatchNorm2d(num_features=output_filters, momentum=bn_mom, eps=bn_eps)
        
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate):
        x = inputs
        if self._expand:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        x_squeezed = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        x_squeezed = self._se_reduce(x_squeezed)
        x_squeezed = self._swish(x_squeezed)
        x_squeezed = self._se_expand(x_squeezed)
        x = torch.sigmoid(x_squeezed) * x

        x = self._project_conv(x)
        x = self._bn2(x)

        if self.id_skip:
            x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs
        return x

class Block(torch.nn.Module):
    def __init__(self,num_repeat, kernel_size, stride, expand_ratio, input_filters, output_filters, se_reduce):
        super().__init__()
        
        self._mb_conv_blocks = torch.nn.ModuleList([])
        self._mb_conv_blocks.append(MBConvBlock(kernel_size, stride, expand_ratio, input_filters, output_filters, se_reduce))
        
        stride=1
        input_filters=output_filters
        
        for _ in range(num_repeat - 1):
            self._mb_conv_blocks.append(MBConvBlock(kernel_size, stride, expand_ratio, input_filters, output_filters, se_reduce))
    
    def forward(self,x,drop_connect_rate):
        for idx, block in enumerate(self._mb_conv_blocks):
            drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
        return x

class EfficientNetBackbone(torch.nn.Module):
    def __init__(self,in_channels,drop_connect_rate=0.2):
        super().__init__()
        bn_mom = 1 - 0.99
        bn_eps = 0.001
        self.drop_connect_rate = drop_connect_rate
        
        out_channels = efficientnet_b4_parameters[0]['input_filters']
        self.stem = torch.nn.Sequential(
            Conv2dDynamicSamePadding(in_channels, out_channels, kernel_size=3, stride=2, bias=False,padding='same'),
            torch.nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps),
            MemoryEfficientSwish()
        )

        self._block0 = Block(**efficientnet_b4_parameters[0]) #256
        self._block1 = Block(**efficientnet_b4_parameters[1]) #128
        self._block2 = Block(**efficientnet_b4_parameters[2]) #64
        self._block3 = Block(**efficientnet_b4_parameters[3]) #32
        self._block4 = Block(**efficientnet_b4_parameters[4]) #32
        self._block5 = Block(**efficientnet_b4_parameters[5]) #16
        self._block6 = Block(**efficientnet_b4_parameters[6]) #16
        
    def forward(self, x): #512
        fms = []
        x = self.stem(x) #256
        fms.append(x)
        x = self._block0(x,drop_connect_rate=self.drop_connect_rate*0/7) #256
        x = self._block1(x,drop_connect_rate=self.drop_connect_rate*1/7) #128
        fms.append(x)
        x = self._block2(x,drop_connect_rate=self.drop_connect_rate*2/7) #64
        fms.append(x)
        x = self._block3(x,drop_connect_rate=self.drop_connect_rate*3/7) #32
        x = self._block4(x,drop_connect_rate=self.drop_connect_rate*4/7) #32
        fms.append(x)
        x = self._block5(x,drop_connect_rate=self.drop_connect_rate*5/7) #16
        x = self._block6(x,drop_connect_rate=self.drop_connect_rate*6/7) #16
        fms.append(x)
        return x

class DecoderBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
             torch.nn.Conv2d(in_channels,out_channels,3,padding='same'),
             torch.nn.BatchNorm2d(in_channels),
             torch.nn.ReLU(inplace=True))
        self.conv2 = torch.nn.Sequential(
             torch.nn.Conv2d(out_channels,out_channels,3,padding='same'),
             torch.nn.BatchNorm2d(out_channels),
             torch.nn.ReLU(inplace=True))

    def forward(self, x, fm=None):
        x = torch.nn.functional.interpolate(x,scale_factor=2,mode='nearest')
        if fm != None:
            x = torch.cat([x, fm], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Decoder(torch.nn.Module):
    def __init__(self,output_channels):
        super().__init__()
        self.block1 = DecoderBlock(448+160,256) #32
        self.block2 = DecoderBlock(256+56,128) #64
        self.block3 = DecoderBlock(128+32,64) #128
        self.block4 = DecoderBlock(64+48,32) #256
        self.block5 = DecoderBlock(32,16) #512
        self.head = torch.nn.Conv2d(16, output_channels, kernel_size=3, padding='same')    
        
    def forward(self,fms):
        fms = fms[::-1]
        x = fms[0]
        x = self.block1(x,fms[1])
        x = self.block2(x,fms[2])
        x = self.block3(x,fms[3])
        x = self.block4(x,fms[4])
        x = self.block5(x)
        x = self.head(x)
        return x

class Unet_EfficientNetB4_reproduce(torch.nn.Module):
    def __init__(self,input_channels,output_channels,drop_connect_rate=0.2):
        super().__init__()
        self.encoder = EfficientNetBackbone(in_channels=input_channels,drop_connect_rate=drop_connect_rate)
        self.decoder = Decoder(output_channels=output_channels)
        
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def freeze(self):
        pass


# class AttentionDownSample(torch.nn.Module):
#     def __init__(self,downscale,in_channel,reduce):
#         super().__init__()
#         self.downscale = downscale
#         self.Q = torch.nn.Linear(in_channel, in_channel//reduce, bias=False)
#         self.K = torch.nn.Linear(in_channel, in_channel//reduce, bias=False)
        
#     def forward(self,fm):
#         B,C,H,W = fm.shape
#         new_H = H//self.downscale
#         new_W = W//self.downscale
        
#         fm = torch.reshape(fm,[B, C, new_H, self.downscale, new_W, self.downscale])
#         fm = torch.permute(fm,[0,2,4,3,5,1])
#         fm = torch.reshape(fm,[B, new_H, new_W, self.downscale*self.downscale, C])
        
#         q = torch.mean(fm,axis=-2,keepdim=True)
#         q = self.Q(q)
#         q = q / q.shape[-1]**0.5
        
#         k = self.K(fm)
        
#         qk = torch.matmul(q,torch.transpose(k,-2,-1))
#         qk = torch.softmax(qk, dim=-1)
        
#         out = torch.matmul(qk,fm)
#         out = torch.squeeze(out,dim=-2)
#         out = torch.permute(out,[0,3,1,2])
#         return out
    
# class Stem(torch.nn.Module):
#     def __init__(self,input_channels,start_chennels):
#         super().__init__()
#         self.body = torch.nn.Sequential(
#             torch.nn.Conv2d(input_channels,64,7,padding='same',bias=False),
#             AttentionDownSample(2,64,2),
#             torch.nn.BatchNorm2d(64),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.MaxPool2d(3,2, padding='same'))
            
            
#         self.merge = torch.nn.Conv2d(8,8,3,padding='same')
    
#     def forward(self,image):
#         x = self.body(image)
#         x = torch.cat((x,image),dim=1)
#         x = self.merge(x)
#         return x
    
# class SubBlock(torch.nn.Sequential):
#     def __init__(self,in_channel,reduce):
#         super().__init__(
#             torch.nn.Conv2d(in_channel,in_channel//reduce,1,bias=False),
#             torch.nn.Conv2d(in_channel//reduce,in_channel//reduce,3,padding='same',groups=in_channel//reduce),
#             torch.nn.InstanceNorm2d(in_channel//reduce, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False),
#             torch.nn.Mish(inplace=True),
#             torch.nn.Conv2d(in_channel//reduce,in_channel,1),
#             torch.nn.InstanceNorm2d(in_channel, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False),
#             torch.nn.Mish(inplace=True)
#             )
    
# class ResBlock(torch.nn.Module):
#     def __init__(self,n_subblocks,in_channel,reduce):
#         super().__init__()
#         if not isinstance(in_channel,(list,tuple)):
#             assert isinstance(in_channel, int)
#             in_channel = [in_channel] * n_subblocks
#         if not isinstance(reduce,(list,tuple)):
#             assert isinstance(reduce, int)
#             reduce = [reduce] * n_subblocks
#         assert len(in_channel) == n_subblocks
#         assert len(reduce) == n_subblocks
#         self.fn = torch.nn.Sequential(*[SubBlock(in_channel[i],reduce[i]) for i in range(n_subblocks)])
#     def forward(self,inp):
#         x = self.fn(inp)
#         x = (x + inp)/2
#         return x
    
# class AttentionBlock(torch.nn.Module):
#     def __init__(self,n_resblocks,n_subblocks,in_channel,reduce):
#         super().__init__()
#         self.fn = torch.nn.Sequential(*[ResBlock(n_subblocks,in_channel,reduce) for _ in range(n_resblocks)])
#         self.att_inp = torch.nn.Conv2d(in_channel,in_channel//reduce,1,bias=False)
#         self.att_x = torch.nn.Conv2d(in_channel,in_channel//reduce,1,bias=False)
        
#     def forward(self,inp):
#         x = self.fn(inp)
#         att = torch.sigmoid(torch.mean(self.att_inp(inp)*self.att_x(x),dim=1,keepdim=True))
#         x = x*att + inp*(1-att)
#         return x

# class Encoder(torch.nn.Module):
#     def __init__(self,input_channels):
#         super().__init__()
#         self.stem = Stem(input_channels,8)
#         self.block0 = torch.nn.Sequential(
#             AttentionDownSample(downscale=2,in_channel=8,reduce=2),
#             torch.nn.Conv2d(8,16,1),
#             torch.nn.InstanceNorm2d(16, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False),
#             torch.nn.Mish(inplace=True),
#             AttentionBlock(n_resblocks=2,n_subblocks=2,in_channel=16,reduce=2),
#             AttentionBlock(n_resblocks=2,n_subblocks=2,in_channel=16,reduce=2)) #512
        
#         self.block1 = torch.nn.Sequential(
#             AttentionDownSample(downscale=4,in_channel=16,reduce=2),
#             torch.nn.Conv2d(16,64,1,bias=False),
#             torch.nn.Mish(inplace=True),
#             AttentionBlock(n_resblocks=2,n_subblocks=2,in_channel=64,reduce=2),
#             AttentionBlock(n_resblocks=2,n_subblocks=2,in_channel=64,reduce=2)) #128
        
#         self.block2 = torch.nn.Sequential( 
#             AttentionDownSample(downscale=4,in_channel=64,reduce=2),
#             torch.nn.Conv2d(64,256,1,bias=False),
#             torch.nn.Mish(inplace=True),
#             AttentionBlock(n_resblocks=2,n_subblocks=3,in_channel=256,reduce=2)) #32
        
#         self.block3 = torch.nn.Sequential(
#             AttentionDownSample(downscale=4,in_channel=256,reduce=2),
#             torch.nn.Conv2d(256,512,1,bias=False),
#             torch.nn.Mish(inplace=True),
#             AttentionBlock(n_resblocks=2,n_subblocks=1,in_channel=512,reduce=2)) #8
        
#     def forward(self,x):
#         fms = []
#         x = self.stem(x)
#         x = self.block0(x) 
#         fms.append(x)
#         x = self.block1(x)
#         fms.append(x)
#         x = self.block2(x)
#         fms.append(x)
#         x = self.block3(x)
#         fms.append(x)
#         return fms
    

# class DecoderBlock(torch.nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         out_channels):
#         super().__init__()
#         self.conv1 = torch.nn.Sequential(
#              torch.nn.Conv2d(in_channels,out_channels,3,padding='same'),
#              torch.nn.Mish(inplace=True))
#         self.conv2 = torch.nn.Sequential(
#              torch.nn.Conv2d(out_channels,out_channels,3,padding='same'),
#              torch.nn.Mish(inplace=True))

#     def forward(self, x, fm=None):
#         x = torch.nn.functional.interpolate(x,scale_factor=2,mode='bilinear',align_corners=True)
#         if fm != None:
#             x = torch.cat([x, fm], dim=1)
#         x = self.conv1(x)
#         x = self.conv2(x)
#         return x

        
# class Decoder(torch.nn.Module):
#     def __init__(self,output_channels):
#         super().__init__()
#         self.upsample1 = DecoderBlock(512,256) #16
#         self.upsample2 = DecoderBlock(256+256,256) #32
#         self.upsample3 = DecoderBlock(256,64) #64
#         self.upsample4 = DecoderBlock(64+64,64) #128
#         self.upsample5 = DecoderBlock(64,16) #256
#         self.upsample6 = DecoderBlock(16+16,output_channels) #512
        
#     def forward(self,fms):
#         fms = fms[::-1]
#         x = fms[0]
#         x = self.upsample1(x)
#         x = self.upsample2(x,fms[1])
#         x = self.upsample3(x)
#         x = self.upsample4(x,fms[2])
#         x = self.upsample5(x)
#         x = self.upsample6(x,fms[3])
#         return x

# class Backbone4(torch.nn.Module):
#     def __init__(self,input_channels,output_channels):
#         super().__init__()
#         self.encoder = Encoder(input_channels=input_channels)
#         self.decoder = Decoder(output_channels=output_channels)
        
#     def forward(self,x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x
    
#     def freeze(self):
#         pass