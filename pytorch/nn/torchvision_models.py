import torch
from torchvision import models

class FCN_ResNet(torch.nn.Module):
    def __str__(self):
        return '.'.join([__name__,self.__class__.__name__])+'('+','.join([f'{k}={v}' for k,v in self.hyperparam_dict.items()]) + ')'
    def __init__(
        self,
        pretrained,
        num_classes,
        trainable_backbone,
        main_head,
        aux_head):
        super().__init__()
        self.hyperparam_dict = dict(
            pretrained = pretrained,
            num_classes = num_classes,
            trainable_backbone = trainable_backbone,
            main_head = main_head,
            aux_head = aux_head)
        
        if aux_head is not None:
            aux_loss = True
        
        self.backbone = models.segmentation.fcn_resnet50(pretrained=pretrained,num_classes=num_classes,aux_loss=aux_loss)
        if not trainable_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
            for param in self.backbone.classifier[-1].parameters():
                param.requires_grad = True
                
            if aux_head is not None:
                for param in self.backbone.aux_classifier[-1].parameters():
                    param.requires_grad = True
        
        self.main_head = main_head
        self.aux_head = aux_head
    
    def forward(self, x):
        x = self.backbone(x)
        out = x['out']
        out = self.main_head(out)
        if self.aux_head is not None:
            aux = x['aux']
            aux = self.aux_head(aux)
            return out, aux
        else:
            return out
    
        
        