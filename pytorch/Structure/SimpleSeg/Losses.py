from utils.loss import BaseLoss

import torch
import torch.nn.functional as F

class DiceLoss(BaseLoss):
    def __init__(self, smoothing=1, reduce='mean', output_numpy=False):
        super().__init__(
            reduce=reduce,
            output_numpy=output_numpy,
            smoothing=smoothing)
        self.smoothing = smoothing

    def call(self,predict,label):
        assert len(predict.shape) == 3
        intersection = (predict * label).sum(axis=[1,2])
        dice = (2.*intersection + self.smoothing)/(predict.sum(axis=[1,2]) + label.sum(axis=[1,2]) + self.smoothing)
        return 1 - dice

class DiceBCELoss(BaseLoss):
    def __init__(self,  smoothing=1, use_logit=True, reduce='mean', output_numpy=False):
        super().__init__(
            reduce=reduce,
            output_numpy=output_numpy,
            smoothing=smoothing)
        self.use_logit = use_logit
        self.smoothing = smoothing

    def call(self, predict, label):
        if self.use_logit:
            BCE_loss = F.binary_cross_entropy_with_logits(predict,label,reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(predict, label, reduction='none')
        BCE_loss = BCE_loss.mean(axis=[1,2])
        
        if self.use_logit:
            predict = torch.sigmoid(predict)
        intersection = (predict * label).sum(axis=[1,2])
        dice_loss = (2.*intersection + self.smoothing)/(predict.sum(axis=[1,2]) + label.sum(axis=[1,2]) + self.smoothing)
        
        Dice_BCE = BCE_loss + dice_loss
        return Dice_BCE
