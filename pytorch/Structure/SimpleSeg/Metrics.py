from utils.metric import BaseMetrics
from . import Losses

class DiceLoss(BaseMetrics):
    def __init__(self,use_logit,smoothing=1):
        super().__init__(
            min,
            Losses.DiceLoss,
            use_logit=use_logit,
            smoothing=smoothing)
        
class DiceBCELoss(BaseMetrics):
    def __init__(self,use_logit,smoothing=1,w_dice=1, w_bce=1):
        super().__init__(
            min,
            Losses.DiceBCELoss,
            w_dice=w_dice,
            w_bce=w_bce,
            use_logit=use_logit,
            smoothing=smoothing)