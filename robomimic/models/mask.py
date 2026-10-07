import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

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


class AttentionGate(nn.Module):
    def __init__(self, query_dim, key_dim, embed_dim):
        super().__init__()
        self.query_proj = nn.Sequential(
            nn.Conv2d(query_dim, embed_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(embed_dim)
        )
        self.key_proj = nn.Sequential(
            nn.Conv2d(key_dim, embed_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(embed_dim)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, query, key):
        # query: decoder features
        # key: encoder skip connections
        query_embed = self.query_proj(query)
        key_embed = self.key_proj(key)

        # ALIGN BEFORE ADD
        query_embed = match_spatial(query_embed, key_embed)

        # Combine query and key signals to compute attention mask
        psi_embed = self.relu(query_embed + key_embed)
        psi_embed = self.psi(psi_embed)

        # ALIGN BEFORE MULTIPLY
        psi_embed = match_spatial(psi_embed, key)
        # Return the 'attended' skip connection
        return psi_embed*key


class SoftAttentionMask(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.attn3 = AttentionGate(query_dim=256, key_dim=256, embed_dim=128)
        self.dec3 = conv_block(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.attn2 = AttentionGate(query_dim=128, key_dim=128, embed_dim=64)
        self.dec2 = conv_block(256,128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.attn1 = AttentionGate(query_dim=64, key_dim=64, embed_dim=32)
        self.dec1 = conv_block(128, 64)

        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)  #64
        e2 = self.enc2(self.pool(e1))  #128
        e3 = self.enc3(self.pool(e2))  #256

        b = self.bottleneck(self.pool(e3))  # 512

        # Level 3 attention and decode
        d3 = self.up3(b)
        e3_attn = self.attn3(query=d3, key=e3)
        e3_attn = match_spatial(e3_attn, d3)

        d3 = self.dec3(torch.cat([d3, e3_attn], dim=1))

        # Level 2 attention and decode
        d2 = self.up2(d3)
        e2_attn = self.attn2(query=d2, key=e2)
        e2_attn = match_spatial(e2_attn, d2)
        d2 = self.dec2(torch.cat([d2, e2_attn], dim=1))

        # Level 1 attention and decode
        d1 = self.up1(d2)
        e1_attn = self.attn1(query=d1, key=e1)
        e1_attn = match_spatial(e1_attn, d1)
        d1 = self.dec1(torch.cat([d1, e1_attn], dim=1))

        out = torch.sigmoid(self.out(d1))
        # out = self.out(d1)
        return out