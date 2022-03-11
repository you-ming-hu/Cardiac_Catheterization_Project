from utils.model import BaseBackbone
import torchvision as tv
import segmentation_models_pytorch as smp

class torchvision_FCN_ResNet(BaseBackbone):
    def __init__(self, pretrained, out_dim):
        
        stem = tv.models.segmentation.fcn_resnet50(pretrained=pretrained,num_classes=out_dim,aux_loss=None)
        non_frozen_layers = [stem.backbone.con1, stem.backbone.bn1, stem.classifier[-1]]
        
        super().__init__(stem,non_frozen_layers)
        
    def forward(self, x):
        x = self.stem(x)
        return x
    
class smp_Unet(BaseBackbone):
    def __init__(self, encoder_name, encoder_weights, in_channels, activation, output_dim):
        
        stem = smp.Unet(
            encoder_name = encoder_name, 
            encoder_weights = encoder_weights, 
            in_channels = in_channels, 
            classes = output_dim,
            activation = activation,
            aux_params = None)
        non_froozen_layers = [stem.encoder.conv1, stem.encoder.bn1, stem.segmentation_head]
        
        super().__init__(stem,non_froozen_layers)
        
    def forward(self, x):
        x = self.stem(x)
        return x