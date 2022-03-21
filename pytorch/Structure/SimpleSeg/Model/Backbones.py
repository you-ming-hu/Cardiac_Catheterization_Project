from utils.model import BaseBackbone
import torchvision as tv
import segmentation_models_pytorch as smp

from .Customs.Backbone1 import Backbone1
from .Customs.Backbone2 import Backbone2

class torchvision_FCN_ResNet(BaseBackbone):
    def __init__(self, pretrained, out_dim):
        stem = tv.models.segmentation.fcn_resnet50(pretrained=False,num_classes=out_dim,aux_loss=None,pretrained_backbone=pretrained)
        non_frozen_layers = [stem.backbone.conv1, stem.backbone.bn1, stem.classifier[-1]]
        
        super().__init__(stem,non_frozen_layers)
        
    def forward(self, x):
        x = self.stem(x)
        x = x['out']
        return x
    
class smp_Unet(BaseBackbone):
    def __init__(self, encoder_name, encoder_weights, in_channels, output_dim):
        stem = smp.Unet(
            encoder_name = encoder_name, 
            encoder_weights = encoder_weights, 
            in_channels = in_channels, 
            classes = output_dim,
            activation = None,
            aux_params = None)
        if encoder_name.lower().startswith('resnet'):
            non_froozen_layers = [stem.encoder.conv1, stem.encoder.bn1, stem.segmentation_head]
        elif encoder_name.lower().startswith('mobilenet'):
            non_froozen_layers = [stem.encoder.features[0][0], stem.encoder.features[0][1], stem.segmentation_head]
        elif encoder_name.lower().startswith('efficientnet'):
            non_froozen_layers = [stem.encoder._conv_stem, stem.encoder._bn0, stem.segmentation_head]
        else:
            assert False
        super().__init__(stem,non_froozen_layers)
        
    def forward(self, x):
        x = self.stem(x)
        return x