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
    def __init__(self):
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Conv2d(1,4,7,padding='same'),
            torch.nn.Mish(inplace=True),
            torch.nn.Conv2d(4,7,3,padding='same'),
            torch.nn.Mish(inplace=True)
            )
        self.merge = torch.nn.Conv2d(8,8,3,padding='same')
    
    def forward(self,image):
        x = self.body(image)
        x = torch.cat((x,image),dim=1)
        x = self.merge(x)
        return x
    
class ResBlock(torch.nn.Module):
    def __init__(self,in_channel,reduce,n_subblocks):
        super().__init__()
        self.fn = torch.nn.Sequential(*self.create_subblocks(in_channel,reduce,n_subblocks))
        self.att_inp = torch.nn.Conv2d(in_channel,in_channel//reduce,1,bias=False)
        self.att_x = torch.nn.Conv2d(in_channel,in_channel//reduce,1,bias=False)
    
    def create_subblocks(self,in_channel,reduce,n):
        return [torch.nn.Sequential(
            torch.nn.Conv2d(in_channel,in_channel//reduce,1,bias=False),
            torch.nn.Conv2d(in_channel//reduce,in_channel//reduce,3,padding='same',groups=in_channel//reduce),
            torch.nn.Mish(inplace=True),
            torch.nn.Conv2d(in_channel//reduce,in_channel,1),
            torch.nn.Mish(inplace=True),
            ) for _ in range(n)]
        
    def forward(self,inp):
        x = self.fn(inp)
        att = torch.sigmoid(torch.mean(self.att_inp(inp)*self.att_x(x),dim=1,keepdim=True))
        x = x*att + inp*(1-att)
        return x

class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = Stem() # 512*512 # 8
        self.block0 = ResBlock(in_channel=8,reduce=2,n_subblocks=2) # 512*512 # 8
        
        self.block1 = torch.nn.Sequential( # 64*64 # 16
            AttentionDownSample(downscale=8,in_channel=8,reduce=2),
            torch.nn.Conv2d(8,16,1,bias=False),
            torch.nn.Mish(inplace=True),
            ResBlock(in_channel=16,reduce=2,n_subblocks=2)
            )
        self.fms_reduce_1 = torch.nn.Conv2d(16, 4, 1, bias=False)
        
        self.block2 = torch.nn.Sequential( # 16*16 # 32
            AttentionDownSample(downscale=4,in_channel=16,reduce=2),
            torch.nn.Conv2d(16,32,1,bias=False),
            torch.nn.Mish(inplace=True),
            ResBlock(in_channel=32,reduce=4,n_subblocks=2)
            )
        self.fms_reduce_2 = torch.nn.Conv2d(32, 4, 1, bias=False)
        
        self.block3 = torch.nn.Sequential( # 8*8 # 64
            AttentionDownSample(downscale=2,in_channel=32,reduce=2),
            torch.nn.Conv2d(32,64,1,bias=False),
            torch.nn.Mish(inplace=True),
            ResBlock(in_channel=64,reduce=4,n_subblocks=2)
            )
        self.fms_reduce_3 = torch.nn.Conv2d(64, 4, 1, bias=False)
        
    def forward(self,x):
        fms = []
        x = self.stem(x)
        x = self.block0(x)
        x = self.block1(x)
        fms.append(self.fms_reduce_1(x))
        x = self.block2(x)
        fms.append(self.fms_reduce_2(x))
        x = self.block3(x)
        fms.append(self.fms_reduce_3(x))
        return fms
        
        
class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Conv2d(4+4+4, 4, 1),
            torch.nn.Mish(inplace=True)
            )
        
    def forward(self,fms):
        for i in range(len(fms)):
            fms[i] = torch.nn.functional.interpolate(fms[i],size=(512,512),mode='bilinear',align_corners=True)
        fms = torch.cat(fms,dim=1)
        fms = self.body(fms)
        return fms

class Backbone1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def freeze(self):
        pass