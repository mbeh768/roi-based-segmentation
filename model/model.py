import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from base import BaseModel


class _DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResNetSegmentation(BaseModel):
    """
    ImageNet-pretrained ResNet-50 encoder + simple convolutional decoder
    for dense segmentation.

    Encoder: ResNet-50 without avgpool/fc → (B, 2048, H/32, W/32)
    Decoder: 5× bilinear upsample + conv blocks  → (B, num_classes, H, W)
    Output:  raw logits — use BCEWithLogitsLoss for num_classes=1,
             CrossEntropyLoss for num_classes>1

    Config:
        "arch": {"type": "ResNetSegmentation", "args": {"num_classes": 1}}
    """
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()

        backbone = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)

        # Encoder: strip the classification head, keep spatial layers
        self.encoder = nn.Sequential(
            backbone.conv1,    # → (B,   64, H/2,  W/2)
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,  # → (B,   64, H/4,  W/4)
            backbone.layer1,   # → (B,  256, H/4,  W/4)
            backbone.layer2,   # → (B,  512, H/8,  W/8)
            backbone.layer3,   # → (B, 1024, H/16, W/16)
            backbone.layer4,   # → (B, 2048, H/32, W/32)
        )

        self.decoder = nn.Sequential(
            _DecoderBlock(2048, 1024),  # → H/16
            _DecoderBlock(1024,  512),  # → H/8
            _DecoderBlock( 512,  256),  # → H/4
            _DecoderBlock( 256,  128),  # → H/2
            _DecoderBlock( 128,   64),  # → H
        )

        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return self.head(x)


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
