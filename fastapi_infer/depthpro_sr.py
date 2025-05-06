import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DepthProForDepthEstimation

class DepthProForSuperResolution(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.base_model = DepthProForDepthEstimation(config)
        hidden_size = config.fusion_hidden_size

        self.image_head = nn.Sequential(
            nn.ConvTranspose2d(config.num_channels, hidden_size, 4, 2, 1),
            nn.ReLU()
        )

        self.head = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_size, hidden_size, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, config.num_channels, 3, 1, 1)
        )

    def forward(self, x):
        encoder_features = self.base_model.depth_pro(x).features
        fused = self.base_model.fusion_stage(encoder_features)[-1]
        x = self.image_head(x)
        x = F.interpolate(x, size=fused.shape[2:])
        x = x + fused
        return self.head(x)
