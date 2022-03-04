import torch
import segmentation_models_pytorch as smp

class Unet(torch.nn.Module):
    def __str__(self):
        return '.'.join([__name__,self.__class__.__name__])+'('+','.join([f'{k}={v}' for k,v in self.hyperparam_dict.items()]) + ')'
    def __init__(
        self,
        encoder_name,
        encoder_weights,
        in_channels,
        activation,
        segmentation_classes,
        segmentation_head,
        classfy_classes,
        classfy_head,
        trainable_backbone):
        
        super().__init__()
        self.hyperparam_dict = dict(
            encoder_name = encoder_name,
            encoder_weights = encoder_weights,
            in_channels = in_channels,
            activation = activation,
            segmentation_classes = segmentation_classes,
            segmentation_head = segmentation_head,
            classfy_classes = classfy_classes,
            classfy_head = classfy_head,
            trainable_backbone = trainable_backbone)
        
        if classfy_head != None:
            classfy_params = {'classes':classfy_classes}
        else:
            classfy_params = None
            
        self.backbone = smp.Unet(
            encoder_name = encoder_name, 
            encoder_weights = encoder_weights, 
            in_channels = in_channels, 
            classes = segmentation_classes,
            activation = activation,
            aux_params = classfy_params)
        
        if not trainable_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
            for param in self.backbone.segmentation_head.parameters():
                param.requires_grad = True
                
            if classfy_head != None:
                for param in self.backbone.classification_head.parameters():
                    param.requires_grad = True
        
        self.segmentation_head = segmentation_head
        self.classfy_head = classfy_head
        
    def forwrd(self,x):
        if self.classfy_head != None:
            segmentation, classify = self.backbone(x)
            segmentation = self.segmentation_head(segmentation)
            classify = self.classfy_head(classify)
            return segmentation, classify
        else:
            segmentation = self.backbone(x)
            segmentation = self.segmentation_head(segmentation)
            return segmentation