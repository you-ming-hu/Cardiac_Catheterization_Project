from utils.metric import BaseMetrics
from . import Losses

class DiceLoss(BaseMetrics):
    def __init__(self,smoothing=1):
        super().__init__(
            min,
            Losses.DiceLoss,
            smoothing=smoothing)
        
class DiceBCELoss(BaseMetrics):
    def __init__(self,use_logit,smoothing=1):
        super().__init__(
            min,
            Losses.DiceBCELoss,
            use_logit=use_logit,
            smoothing=smoothing)