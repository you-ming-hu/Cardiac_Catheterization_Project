import torch



class AttentionDownSample(torch.nn.Module):
    def __init__(self,fm_shape,scale,in_channel,reduce):
        super().__init__()
        self.h,self.w = fm_shape
        self.new_h, self.new_w = self.h//scale, self.w//scale
        self.scale = scale
        self.in_channel = in_channel
        
        self.K = torch.nn.Linear(in_channel, in_channel//reduce, bias=False)
        self.Q = torch.nn.Linear(in_channel, in_channel//reduce, bias=False)
        self.V = torch.nn.Linear(in_channel, in_channel//reduce, bias=False)
        
    def forward(self,fm):
        fm = torch.reshape(fm,[-1, self.in_channel, self.new_h, self.scale, self.new_w, self.scale])
        fm = torch.permute(fm,[0,2,4,3,5,1])
        fm = torch.reshape(fm,[-1, self.new_h, self.new_w, self.scale*self.scale, self.in_channel])
        
        k = torch.mean(fm,axis=-2,keepdim=True)
        k = self.K(k)
        
        q = self.Q(fm)
        v = self.V(fm)
        
        qk = torch.matmul(k,torch.transpose(q,-2,-1))
        qk = qk / torch.sqrt(float(q.shape[-1]))
        qk = torch.softmax(qk, dim=-1)
        
        out = torch.matmul(qk,v)
        out = torch.squeeze(out,dim=-2)
        out = torch.permute(out,[0,3,1,2])
        return out