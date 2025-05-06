# model.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics.audio import SignalNoiseRatio
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from transformers import DepthProConfig, DepthProForDepthEstimation

class DepthProForSuperResolution(nn.Module):
    def __init__(self, depthpro_for_depth_estimation):
        super(DepthProForSuperResolution, self).__init__()
        self.depthpro_for_depth_estimation = depthpro_for_depth_estimation
        hidden_size = self.depthpro_for_depth_estimation.config.fusion_hidden_size

        self.image_head = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=depthpro_for_depth_estimation.config.num_channels,
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
            nn.Conv2d(hidden_size, depthpro_for_depth_estimation.config.num_channels, 3, 1, 1),
        )

    def forward(self, x):
        encoder_features = self.depthpro_for_depth_estimation.depth_pro(x).features
        fused_hidden_state = self.depthpro_for_depth_estimation.fusion_stage(encoder_features)[-1]
        x = self.image_head(x)
        x = F.interpolate(x, size=fused_hidden_state.shape[2:])
        x = x + fused_hidden_state
        x = self.head(x)
        return x

class LightningModel(L.LightningModule):
    def __init__(self, depthpro_for_depth_estimation):
        super().__init__()
        self.model = DepthProForSuperResolution(depthpro_for_depth_estimation)
        self.snr = SignalNoiseRatio()
        self.psnr = PeakSignalNoiseRatio()
        self.ssim = StructuralSimilarityIndexMeasure()
        self.mse = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        lr, hr = batch
        sr = self.model(lr)
        sr = F.interpolate(sr, size=hr.shape[2:])
        loss = self.mse(sr, hr)

        self.log("train_mse_loss", loss, prog_bar=True)
        self.log("train_mean_grad_norm", self.log_gradient_norms(), prog_bar=True)
        self.log("train_lr", self.optimizers().param_groups[0]['lr'], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        lr, hr = batch
        sr = self.model(lr)
        sr = F.interpolate(sr, size=hr.shape[2:])

        mse_loss = self.mse(sr, hr)
        psnr_value = self.psnr(sr, hr)
        ssim_value = self.ssim(sr, hr)
        snr_value = self.snr(sr, hr)

        self.log("val_mse_loss", mse_loss, prog_bar=True)
        self.log("val_psnr", psnr_value, prog_bar=True)
        self.log("val_ssim", ssim_value, prog_bar=True)
        self.log("val_snr", snr_value, prog_bar=True)

        # Save SR image
        if batch_idx == 0 and hasattr(self.logger, "save_dir"):
            sr_image = sr[0].detach().cpu().permute(1, 2, 0).float().numpy()
            sr_image = sr_image * 0.5 + 0.5
            sr_image = sr_image.clip(0.0, 1.0)
            save_dir = os.path.join(self.logger.save_dir, self.logger.name, f"version_{self.logger.version}", "sr_images")
            os.makedirs(save_dir, exist_ok=True)
            sr_image_save_path = os.path.join(save_dir, f"{self.global_step}.png")
            import matplotlib.pyplot as plt
            plt.matshow(sr_image)
            plt.axis('off')
            plt.savefig(sr_image_save_path)
            plt.close()

        return mse_loss

    def test_step(self, batch, batch_idx):
        lr, hr = batch
        sr = self.model(lr)
        sr = F.interpolate(sr, size=hr.shape[2:])

        self.log("test_mse_loss", self.mse(sr, hr))
        self.log("test_psnr", self.psnr(sr, hr))
        self.log("test_ssim", self.ssim(sr, hr))
        self.log("test_snr", self.snr(sr, hr))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
        return [optimizer], [scheduler]

    def log_gradient_norms(self):
        total_grad_norm = 0.0
        num_params = 0
        for param in self.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm(2).item()
                num_params += 1
        return total_grad_norm / num_params if num_params > 0 else 0

def get_depthpro_model(args):
    patch_size = args.patch_size
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

