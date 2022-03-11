from utils.model import BaseBackbone
import torchvision as tv

class torchvision_FCN_ResNet(BaseBackbone):
    def __init__(
        self,
        pretrained,
        num_classes):
        
        stem = tv.models.segmentation.fcn_resnet50(pretrained=pretrained,num_classes=num_classes,aux_loss=True)
        non_frozen_layers = [stem.backbone.con1, stem.backbone.bn1,stem.classifier[-1],stem.aux_classifier[-1]]
        
        super().__init__(stem,non_frozen_layers)
    
    def forward(self, x):
        output = self.stem(x)
        out = output['out']
        aux = output['aux']
        return out, aux