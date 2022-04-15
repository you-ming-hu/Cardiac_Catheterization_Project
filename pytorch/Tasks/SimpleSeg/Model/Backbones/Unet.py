
import segmentation_models_pytorch as smp
from Base.Model import BaseBackbone

class Unet(BaseBackbone):
    def __init__(self, encoder_name, encoder_weights, in_channels, out_channels):
        stem = smp.Unet(
            encoder_name = encoder_name, 
            encoder_weights = encoder_weights, 
            in_channels = in_channels, 
            classes = out_channels,
            activation = None,
            aux_params = None)
        if encoder_name.lower().startswith('resnet'):
            non_froozen_layers = [stem.encoder.conv1, stem.encoder.bn1, stem.segmentation_head]
        elif encoder_name.lower().startswith('mobilenet'):
            non_froozen_layers = [stem.encoder.features[0][0], stem.encoder.features[0][1], stem.segmentation_head]
        elif encoder_name.lower().startswith('efficientnet'):
            non_froozen_layers = [stem.encoder._conv_stem, stem.encoder._bn0, stem.segmentation_head]
        elif encoder_name.lower().startswith('densenet'):
            non_froozen_layers = [stem.encoder.features[0], stem.encoder.features[1], stem.segmentation_head]
        else:
            assert False
        super().__init__(stem,non_froozen_layers)
        
    def forward(self, x):
        x = self.stem(x)
        return x