from src.metrics.base_metric import BaseMetric
from .wv_mos import Wav2Vec2MOS
from torch import Tensor
from src.utils.load_wvmos import download_mos
import numpy as np

class MOSMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        download_mos()
        self.mos_meter = Wav2Vec2MOS(path='.cache/wv_mos/wv_mos.ckpt')
        self.values = []

    def __call__(self, audio_pred: Tensor, **batch):
        '''audios: [B, 1, len]'''
        for pred in audio_pred:
            mos = self.mos_meter.calculate_one(pred)
            self.values.append(mos)
        return
    
    def get_mean(self):
        mean = np.mean(self.values)
        self.values = []
        return mean