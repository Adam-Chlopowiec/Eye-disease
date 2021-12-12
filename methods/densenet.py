from typing import Dict
import torch
from torch import nn
from torchvision.models import DenseNet as DNet

from methods.base_model import BaseModel


class DenseNet(BaseModel):

    def __init__(self, num_classes: int):
        super(DenseNet, self).__init__(num_classes)
        self._feature_extractor = DNet(True)
        net_fc = self._feature_extractor.fc.in_features
        self._feature_extractor.fc = nn.Linear(net_fc, self.num_classes)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._feature_extractor(x['input'])
        return out
