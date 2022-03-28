
import torch
import segmentation_models_pytorch as smp
from utils.model import BaseBackbone


class SMP_Unet_EffiNetB4_Non_Pretrain(BaseBackbone):
    def __init__(self, in_channels, out_channels):
        stem = smp.Unet(
            encoder_name = 'efficientnet-b4', 
            encoder_weights = 'imagenet', 
            in_channels = in_channels, 
            classes = out_channels,
            activation = None,
            aux_params = None)
        for param in stem.parameters():
            param.reset_parameters()
        
        non_froozen_layers = [
            stem.encoder._conv_stem,
            stem.encoder._bn0,
            stem.segmentation_head]
        super().__init__(stem,non_froozen_layers)
        
    def forward(self, x):
        x = self.stem(x)
        return x