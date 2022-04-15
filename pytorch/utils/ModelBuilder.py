import torch

class ModelBuilder(torch.nn.Module):
    def __init__(self,backbone,head):
        super().__init__()
        self.backbone = backbone
        self.head = head
        
    def freeze_backbone(self):
        self.backbone.freeze()
        
    def forward(self,x):
        x = self.backbone(x)
        x = self.head(x)
        return x
    
    def predict(self,x,threshold=None):
        x = self.backbone(x)
        x = self.head.predict(x)
        if threshold != None:
            assert 0 <= threshold <= 1
            x = (x > threshold).type(torch.float32)
        return x