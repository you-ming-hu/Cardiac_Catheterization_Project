from utils.model import BaseBackbone
import segmentation_models_pytorch as smp

class smp_Unet(BaseBackbone):
    def __init__(
        self,
        encoder_name,
        encoder_weights,
        in_channels,
        activation,
        segmentation_dim,
        classfy_dim):
        
        super().__init__()
            
        stem = smp.Unet(
            encoder_name = encoder_name, 
            encoder_weights = encoder_weights, 
            in_channels = in_channels, 
            classes = segmentation_dim,
            activation = activation,
            aux_params = {'classes':classfy_dim})
        non_froozen_layers = [stem.encoder.conv1, stem.encoder.bn1, stem.segmentation_head, stem.classification_head]
        
        super().__init__(stem,non_froozen_layers)
        
    def forwrd(self,x):
        segmentation, classify = self.stem(x)
        return segmentation, classify