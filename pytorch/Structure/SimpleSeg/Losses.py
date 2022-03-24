from sklearn.metrics import brier_score_loss
from utils.loss import BaseLoss,CompoundLoss

import torch
import torch.nn.functional as F

class DiceLoss(BaseLoss):
    def __init__(self, use_logit = True, smoothing=1, reduce='mean', output_numpy=False):
        super().__init__(
            reduce=reduce,
            output_numpy=output_numpy,
            smoothing=smoothing)
        self.use_logit = use_logit
        self.smoothing = smoothing

    def call(self,predict,label):
        assert len(predict.shape) == 3
        if self.use_logit:
            predict = torch.sigmoid(predict)
        intersection = (predict * label).sum(axis=[1,2])
        dice = (2.*intersection + self.smoothing)/(predict.sum(axis=[1,2]) + label.sum(axis=[1,2]) + self.smoothing)
        return 1 - dice
    
class DiceAccuracy(BaseLoss):
    def __init__(self,use_logit = True, threshold = 0.5, smoothing=1, reduce='mean', output_numpy=False):
        assert 0 <= threshold <= 1
        super().__init__(
            reduce=reduce,
            output_numpy=output_numpy,
            threshold=threshold,
            smoothing=smoothing)
        self.use_logit = use_logit
        self.threshold = threshold
        self.smoothing = smoothing
    def call(self,predict,label):
        assert len(predict.shape) == 3
        if self.use_logit:
            predict = torch.sigmoid(predict)
        predict = (predict > self.threshold).type(torch.float32)
        intersection = (predict * label).sum(axis=[1,2])
        dice = (2.*intersection + self.smoothing)/(predict.sum(axis=[1,2]) + label.sum(axis=[1,2]) + self.smoothing)
        return 1 - dice
    
    
class BCELoss(BaseLoss):
    def __init__(self, use_logit=True, reduce='mean', output_numpy=False):
        super().__init__(
            reduce=reduce,
            output_numpy=output_numpy)
        self.use_logit = use_logit
        
    def call(self,predict,label):
        if self.use_logit:
            BCE_loss = F.binary_cross_entropy_with_logits(predict,label,reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(predict, label, reduction='none')
        BCE_loss = BCE_loss.mean(axis=[1,2])
        return BCE_loss

class DiceBCELoss(CompoundLoss):
    def __init__(self, w_dice=1, w_bce=1, use_logit=True,  smoothing=1, reduce='mean', output_numpy=False):
        super().__init__(
            reduce=reduce,
            output_numpy=output_numpy)
        self.add_loss(w_dice,DiceLoss,use_logit=use_logit,smoothing=smoothing)
        self.add_loss(w_bce,BCELoss,use_logit=use_logit)
        