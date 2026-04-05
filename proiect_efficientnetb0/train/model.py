import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from train.config import NUM_CLASSES


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.25, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model
