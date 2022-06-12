# import torch
import segmentation_models_pytorch as smp

class Backbone(smp.Unet):
    def __init__(
        self,
        encoder_name,
        encoder_weights,
        in_channels,
        activation,
        segmentation_dim,
        classfy_dim):
        
        super().__init__(
            encoder_name = encoder_name, 
            encoder_weights = encoder_weights, 
            in_channels = in_channels, 
            classes = segmentation_dim,
            activation = activation,
            aux_params = {'classes':classfy_dim})
            
    def forwrd(self,x):
        seg, classify  = super()(x)
        return {'seg':seg,'classify':classify}
    
    
def create_module(
    encoder_name,
    encoder_weights,
    in_channels,
    activation,
    segmentation_dim,
    classfy_dim):
    return Backbone(
        encoder_name,
        encoder_weights,
        in_channels,
        activation,
        segmentation_dim,
        classfy_dim)