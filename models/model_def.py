
# model_def.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DepthProConfig, DepthProForDepthEstimation

class DepthProForSuperResolution(nn.Module):
    def __init__(self, depthpro_for_depth_estimation):
        super().__init__()
        self.depthpro_for_depth_estimation = depthpro_for_depth_estimation
        hidden_size = self.depthpro_for_depth_estimation.config.fusion_hidden_size

        self.image_head = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=3,
                out_channels=hidden_size,
                kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_size, hidden_size, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, 3, 3, 1, 1),
        )

    def forward(self, x):
        encoder_features = self.depthpro_for_depth_estimation.depth_pro(x).features
        fused_hidden_state = self.depthpro_for_depth_estimation.fusion_stage(encoder_features)[-1]
        x = self.image_head(x)
        x = F.interpolate(x, size=fused_hidden_state.shape[2:])
        x = x + fused_hidden_state
        x = self.head(x)
        return x

def get_depthpro_model(patch_size=32, compressed=False):
    if compressed:
        config = DepthProConfig(
            patch_size=patch_size,
            patch_embeddings_size=4,
            num_hidden_layers=6,
            intermediate_hook_ids=[8, 5],
            intermediate_feature_dims=[128] * 2,
            scaled_images_ratios=[0.5],
            scaled_images_overlap_ratios=[0.5],
            scaled_images_feature_dims=[256],
            use_fov_model=False,
            fusion_hidden_size=256
        )
    else:
        config = DepthProConfig(
            patch_size=patch_size,
            patch_embeddings_size=4,
            num_hidden_layers=12,
            intermediate_hook_ids=[11, 8, 7, 5],
            intermediate_feature_dims=[256] * 4,
            scaled_images_ratios=[0.5, 1.0],
            scaled_images_overlap_ratios=[0.5, 0.25],
            scaled_images_feature_dims=[1024, 512],
            use_fov_model=False,
        )

    model, _ = DepthProForDepthEstimation.from_pretrained(
        "geetu040/DepthPro", revision="project", config=config,
        ignore_mismatched_sizes=True, output_loading_info=True
    )
    return model

__all__ = ["DepthProForSuperResolution", "get_depthpro_model"]