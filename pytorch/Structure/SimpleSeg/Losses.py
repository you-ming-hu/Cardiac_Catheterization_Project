from utils.loss import BaseLoss

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
    def __init__(self,  smoothing=1, reduce='mean', output_numpy=False):
        super().__init__(
            reduce=reduce,
            output_numpy=output_numpy,
            smoothing=smoothing)
        self.smoothing = smoothing

    def call(self, predict, label):
        intersection = (predict * label).sum(axis=[1,2])
        dice_loss = (2.*intersection + self.smoothing)/(predict.sum(axis=[1,2]) + label.sum(axis=[1,2]) + self.smoothing)
        BCE_loss = F.binary_cross_entropy(predict, label, reduction='none')
        BCE_loss = BCE_loss.mean(axis=[1,2])
        Dice_BCE = BCE_loss + dice_loss
        return Dice_BCE
