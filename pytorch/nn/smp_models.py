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
        main_head,
        classfy_classes,
        aux_head,
        trainable_backbone):
        
        super().__init__()
        self.hyperparam_dict = dict(
            encoder_name = encoder_name,
            encoder_weights = encoder_weights,
            in_channels = in_channels,
            activation = activation,
            segmentation_classes = segmentation_classes,
            main_head = main_head,
            classfy_classes = classfy_classes,
            aux_head = aux_head,
            trainable_backbone = trainable_backbone)
        
        
        if aux_head is not None:
            aux_params = {'classes':classfy_classes}
        else:
            aux_params = None
            
        self.backbone = smp.Unet(
            encoder_name = encoder_name, 
            encoder_weights = encoder_weights, 
            in_channels = in_channels, 
            classes = segmentation_classes, 
            activation = activation,
            aux_params = aux_params})
        
        trainable_backbone
        
        self.main_head = main_head
        self.aux_head = aux_head
        
    def forwrd(self,x):
        pass