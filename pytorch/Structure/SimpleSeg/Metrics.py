from utils.metric import BaseMetrics
from . import Losses

class DiceLoss(BaseMetrics):
    def __init__(self,smoothing=1):
        super().__init__(
            min,
            Losses.DiceLoss,
            smoothing=smoothing)