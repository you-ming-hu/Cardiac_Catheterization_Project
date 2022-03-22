from utils.model import BaseBackbone
import torchvision as tv

class torchvision_FCN_ResNet(BaseBackbone):
    def __init__(self, pretrained, out_dim):
        stem = tv.models.segmentation.fcn_resnet50(pretrained=False,num_classes=out_dim,aux_loss=None,pretrained_backbone=pretrained)
        non_frozen_layers = [stem.backbone.conv1, stem.backbone.bn1, stem.classifier[-1]]
        
        super().__init__(stem,non_frozen_layers)
        
    def forward(self, x):
        x = self.stem(x)
        x = x['out']
        return x
    