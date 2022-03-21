import torch

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
    
class Stem(torch.nn.Module):
    def __init__(self,input_channels,start_chennels):
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels,start_chennels//2,7,padding='same'),
            torch.nn.InstanceNorm2d(start_chennels//2, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False),
            torch.nn.Mish(inplace=True),
            torch.nn.Conv2d(start_chennels//2,start_chennels-input_channels,3,padding='same'),
            torch.nn.InstanceNorm2d(start_chennels-input_channels, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False),
            torch.nn.Mish(inplace=True)
            )
        self.merge = torch.nn.Conv2d(8,8,3,padding='same')
    
    def forward(self,image):
        x = self.body(image)
        x = torch.cat((x,image),dim=1)
        x = self.merge(x)
        return x
    
class SubBlock(torch.nn.Sequential):
    def __init__(self,in_channel,reduce):
        super().__init__(
            torch.nn.Conv2d(in_channel,in_channel//reduce,1,bias=False),
            torch.nn.Conv2d(in_channel//reduce,in_channel//reduce,3,padding='same',groups=in_channel//reduce),
            torch.nn.InstanceNorm2d(in_channel//reduce, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False),
            torch.nn.Mish(inplace=True),
            torch.nn.Conv2d(in_channel//reduce,in_channel,1),
            torch.nn.InstanceNorm2d(in_channel, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False),
            torch.nn.Mish(inplace=True)
            )
    
class ResBlock(torch.nn.Module):
    def __init__(self,n_subblocks,in_channel,reduce):
        super().__init__()
        if not isinstance(in_channel,(list,tuple)):
            assert isinstance(in_channel, int)
            in_channel = [in_channel] * n_subblocks
        if not isinstance(reduce,(list,tuple)):
            assert isinstance(reduce, int)
            reduce = [reduce] * n_subblocks
        assert len(in_channel) == n_subblocks
        assert len(reduce) == n_subblocks
        self.fn = torch.nn.Sequential(*[SubBlock(in_channel[i],reduce[i]) for i in range(n_subblocks)])
    def forward(self,inp):
        x = self.fn(inp)
        x = (x + inp)/2
        return x
    
class AttentionBlock(torch.nn.Module):
    def __init__(self,n_resblocks,n_subblocks,in_channel,reduce):
        super().__init__()
        self.fn = torch.nn.Sequential([ResBlock(n_subblocks,in_channel,reduce) for _ in range(n_resblocks)])
        self.att_inp = torch.nn.Conv2d(in_channel,in_channel//reduce,1,bias=False)
        self.att_x = torch.nn.Conv2d(in_channel,in_channel//reduce,1,bias=False)
        
    def forward(self,inp):
        x = self.fn(inp)
        att = torch.sigmoid(torch.mean(self.att_inp(inp)*self.att_x(x),dim=1,keepdim=True))
        x = x*att + inp*(1-att)
        return x

class Encoder(torch.nn.Module):
    def __init__(self,input_channels):
        super().__init__()
        self.stem = Stem(input_channels,8)
        self.block0 = torch.nn.Sequential(
            torch.nn.Conv2d(8,16,1),
            torch.nn.InstanceNorm2d(16, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False),
            torch.nn.Mish(inplace=True),
            AttentionBlock(n_resblocks=2,n_subblocks=2,in_channel=16,reduce=2),
            AttentionBlock(n_resblocks=2,n_subblocks=2,in_channel=16,reduce=2))
        self.fms_reduce_0 = torch.nn.Conv2d(16, 4, 1, bias=False) #(512,512,4)
        
        self.block1 = torch.nn.Sequential(
            AttentionDownSample(downscale=4,in_channel=16,reduce=2),
            torch.nn.Conv2d(16,32,1,bias=False),
            torch.nn.Mish(inplace=True),
            AttentionBlock(n_resblocks=2,n_subblocks=2,in_channel=32,reduce=2),
            AttentionBlock(n_resblocks=2,n_subblocks=2,in_channel=32,reduce=2))
        self.fms_reduce_1 = torch.nn.Conv2d(32, 8, 1, bias=False) #(128,128,8)
        
        self.block2 = torch.nn.Sequential( 
            AttentionDownSample(downscale=4,in_channel=32,reduce=2),
            torch.nn.Conv2d(32,64,1,bias=False),
            torch.nn.Mish(inplace=True),
            AttentionBlock(n_resblocks=2,n_subblocks=3,in_channel=64,reduce=2))
        self.fms_reduce_2 = torch.nn.Conv2d(64, 16, 1, bias=False) # (32,32,16)
        
        self.block3 = torch.nn.Sequential( 
            AttentionDownSample(downscale=4,in_channel=64,reduce=2),
            torch.nn.Conv2d(64,128,1,bias=False),
            torch.nn.Mish(inplace=True),
            AttentionBlock(n_resblocks=2,n_subblocks=1,in_channel=128,reduce=2))
        self.fms_reduce_3 = torch.nn.Conv2d(128, 32, 1, bias=False) # (8,8,32)
        
    def forward(self,x):
        fms = []
        x = self.stem(x)
        x = self.block0(x)
        fms.append(self.fms_reduce_0(x))
        x = self.block1(x)
        fms.append(self.fms_reduce_1(x))
        x = self.block2(x)
        fms.append(self.fms_reduce_2(x))
        x = self.block3(x)
        fms.append(self.fms_reduce_3(x))
        return fms
        
        
class Decoder(torch.nn.Module):
    def __init__(self,output_channels):
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Conv2d(4+8+16+32, output_channels, 1),
            torch.nn.Mish(inplace=True)
            )
        
    def forward(self,fms):
        for i in range(len(fms)):
            fms[i] = torch.nn.functional.interpolate(fms[i],size=(512,512),mode='bilinear',align_corners=True)
        fms = torch.cat(fms,dim=1)
        fms = self.body(fms)
        return fms

class Backbone2(torch.nn.Module):
    def __init__(self,input_channels,output_channels):
        super().__init__()
        self.encoder = Encoder(input_channels=input_channels)
        self.decoder = Decoder(output_channels=output_channels)
        
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def freeze(self):
        pass