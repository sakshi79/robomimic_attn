import torch
import torch.nn as nn
import torch.nn.functional as F


def match_spatial(x, ref):
    """
    Resize x to have same spatial size as ref (H, W)
    """
    if x.shape[-2:] != ref.shape[-2:]:
        x = F.interpolate(
            x,
            size=ref.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
    return x


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)              # 64
        e2 = self.enc2(self.pool(e1))  # 128
        e3 = self.enc3(self.pool(e2))  # 256

        b = self.bottleneck(self.pool(e3))  # 512

        # Level 3 decode (plain skip connection)
        d3 = self.up3(b)
        e3_skip = match_spatial(e3, d3)
        d3 = self.dec3(torch.cat([d3, e3_skip], dim=1))

        # Level 2 decode
        d2 = self.up2(d3)
        e2_skip = match_spatial(e2, d2)
        d2 = self.dec2(torch.cat([d2, e2_skip], dim=1))

        # Level 1 decode
        d1 = self.up1(d2)
        e1_skip = match_spatial(e1, d1)
        d1 = self.dec1(torch.cat([d1, e1_skip], dim=1))

        out = torch.sigmoid(self.out(d1))
        # out = self.out(d1)
        return out


if __name__ == "__main__":
    model = UNet(in_channels=3, out_channels=3)
    for h, w in [(256, 256), (255, 301)]:
        y = model(torch.randn(2, 3, h, w))
        print((h, w), "->", tuple(y.shape))
    print("params:", sum(p.numel() for p in model.parameters()))