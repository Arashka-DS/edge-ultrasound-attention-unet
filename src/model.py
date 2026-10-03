import torch
import torch.nn as nn
import torchvision.models as models


class AttentionGate(nn.Module):
    """Additive Attention Gate to suppress irrelevant background speckle noise."""

    def __init__(self, f_g: int, f_l: int, f_int: int):
        super().__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int),
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ConvBlock(nn.Module):
    """Standard double convolution block for decoder stages."""

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class VascularAttentionUNet(nn.Module):
    """
    Multi-Head Edge Architecture:
    - Backbone: MobileNetV3-Small (lightweight feature extractor).
    - Input: [B, 3, H, W] representing a rolling temporal stack [t-2, t-1, t].
    - Head 1 (Acoustic Contact): Probe on skin vs. acoustic shadow / air.
    - Head 2 (Vessel Presence): Target artery/vein visible in current cross-section.
    - Head 3 (Attention Decoder): Dense pixel segmentation mask.
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()

        # Lightweight MobileNetV3 backbone
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)
        features = backbone.features

        # Hierarchical Encoder Taps
        self.enc0 = features[0:1]   # 16 channels, stride 2 (128x128)
        self.enc1 = features[1:2]   # 16 channels, stride 2 (64x64)
        self.enc2 = features[2:4]   # 24 channels, stride 2 (32x32)
        self.enc3 = features[4:9]   # 48 channels, stride 2 (16x16)
        self.bottleneck = features[9:]  # 576 channels, stride 2 (8x8)

        # Global Average Pooling for classification heads
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Head 1: Acoustic Probe Contact Classifier
        self.head_contact = nn.Sequential(
            nn.Linear(576, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

        # Head 2: Target Vascular Presence Classifier
        self.head_vessel = nn.Sequential(
            nn.Linear(576, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

        # Decoder Stages with Attention Gates
        self.up4 = nn.ConvTranspose2d(576, 48, kernel_size=2, stride=2)
        self.att4 = AttentionGate(f_g=48, f_l=48, f_int=24)
        self.dec4 = ConvBlock(48 + 48, 48)

        self.up3 = nn.ConvTranspose2d(48, 24, kernel_size=2, stride=2)
        self.att3 = AttentionGate(f_g=24, f_l=24, f_int=12)
        self.dec3 = ConvBlock(24 + 24, 24)

        self.up2 = nn.ConvTranspose2d(24, 16, kernel_size=2, stride=2)
        self.att2 = AttentionGate(f_g=16, f_l=16, f_int=8)
        self.dec2 = ConvBlock(16 + 16, 16)

        self.up1 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.att1 = AttentionGate(f_g=16, f_l=16, f_int=8)
        self.dec1 = ConvBlock(16 + 16, 16)

        # Final projection to 256x256
        self.final_up = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.head_seg = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x: torch.Tensor):
        # Encoder forward pass
        e0 = self.enc0(x)          # [B, 16, 128, 128]
        e1 = self.enc1(e0)         # [B, 16, 64, 64]
        e2 = self.enc2(e1)         # [B, 24, 32, 32]
        e3 = self.enc3(e2)         # [B, 48, 16, 16]
        b = self.bottleneck(e3)    # [B, 576, 8, 8]

        # Classification Heads
        pooled = torch.flatten(self.gap(b), 1)
        contact_logit = self.head_contact(pooled)
        vessel_logit = self.head_vessel(pooled)

        # Segmentation Decoder with Skip-Attention Connections
        d4 = self.up4(b)
        x_att4 = self.att4(g=d4, x=e3)
        d4 = self.dec4(torch.cat([x_att4, d4], dim=1))

        d3 = self.up3(d4)
        x_att3 = self.att3(g=d3, x=e2)
        d3 = self.dec3(torch.cat([x_att3, d3], dim=1))

        d2 = self.up2(d3)
        x_att2 = self.att2(g=d2, x=e1)
        d2 = self.dec2(torch.cat([x_att2, d2], dim=1))

        d1 = self.up1(d2)
        x_att1 = self.att1(g=d1, x=e0)
        d1 = self.dec1(torch.cat([x_att1, d1], dim=1))

        out_mask = self.head_seg(self.final_up(d1))

        return contact_logit, vessel_logit, out_mask
