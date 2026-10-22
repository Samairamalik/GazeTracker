"""L2CS-Net network, vendored from https://github.com/Ahmednull/L2CS-Net (l2cs/model.py and the
ResNet50 branch of l2cs/utils.getArch).

    MIT License. Copyright (c) 2022 Ahmed Abdelrahman.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this
    software and associated documentation files, to deal in the Software without restriction,
    subject to the copyright notice and this permission notice being included in all copies.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

Vendored instead of pip-installed because `import l2cs` pulls in a RetinaFace detector (a
separate git dependency that downloads its own weights); we only need the network and crop the
face with MediaPipe. `forward` returns (yaw_logits, pitch_logits), in that order. Note the
upstream `Pipeline.predict_gaze` unpacks them as (pitch, yaw), i.e. swapped.
"""
import math
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models.resnet import Bottleneck


class L2CS(nn.Module):
    def __init__(self, block, layers, num_bins):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_yaw_gaze = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch_gaze = nn.Linear(512 * block.expansion, num_bins)
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)   # vestigial; kept so weights load strictly
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        layers += [block(self.inplanes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x).view(x.size(0), -1)
        return self.fc_yaw_gaze(x), self.fc_pitch_gaze(x)


def build_l2cs(num_bins: int = 90) -> L2CS:
    """ResNet50 L2CS with random weights (the only architecture used here)."""
    return L2CS(Bottleneck, [3, 4, 6, 3], num_bins)


def load_l2cs(weights: str | Path, device: torch.device) -> L2CS:
    """Load pretrained weights without any unsafe deserialization:
    `.safetensors` via the safetensors library (no pickle at all), anything else via
    torch.load(weights_only=True), with no fallback. Keys must match exactly (strict=True).
    Raises RuntimeError with the reason if loading is not possible."""
    weights = Path(weights)
    if not weights.exists():
        raise RuntimeError(f"L2CS weights not found at {weights} (see CITATIONS.md for the source).")
    try:
        if weights.suffix == ".safetensors":
            from safetensors.torch import load_file
            state = load_file(str(weights), device="cpu")
        else:
            state = torch.load(weights, map_location="cpu", weights_only=True)
    except Exception as e:
        raise RuntimeError(f"Safe loading refused {weights}: {e}. "
                           "Not falling back to unsafe loading.") from e
    model = build_l2cs()
    expected, got = set(model.state_dict()), set(state)
    if expected != got:
        raise RuntimeError(f"L2CS weight keys do not match the vendored ResNet50 L2CS: "
                           f"missing {sorted(expected - got)[:10]}, unexpected {sorted(got - expected)[:10]}")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
