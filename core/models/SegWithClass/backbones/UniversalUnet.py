# import torch
import segmentation_models_pytorch as smp
import torch

class Backbone(smp.Unet):
    def __init__(
        self,
        encoder_name,
        encoder_weights,
        in_channels,
        segmentation_dim,
        classfy_dim):
        
        super().__init__(
            encoder_name = encoder_name, 
            encoder_weights = encoder_weights, 
            in_channels = in_channels, 
            classes = segmentation_dim,
            activation = None,
            aux_params = {'classes':classfy_dim})
        self.mish  = torch.nn.Mish()
            
    def forward(self,x):
        seg, classify  = super().forward(x)
        seg = self.mish(seg)
        classify = self.mish(classify)
        return {'seg':seg,'classify':classify}
    
    
def create_module(
    encoder_name,
    encoder_weights,
    in_channels,
    segmentation_dim,
    classfy_dim):
    return Backbone(
        encoder_name,
        encoder_weights,
        in_channels,
        segmentation_dim,
        classfy_dim)