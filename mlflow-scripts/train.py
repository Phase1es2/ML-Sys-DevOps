import os
import argparse
import mlflow
import mlflow.pytorch
import torch
import time
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from model import LightningModel, get_depthpro_model
from dataloader import get_dataloaders

# ✅ 设置 matmul 精度为 medium（A100 推荐）
torch.set_float32_matmul_precision("medium")

def main(args):
    mlflow.set_tracking_uri("http://129.114.24.214:8000")
    mlflow.set_experiment("SuperResolutionDepthPro")
    mlflow.autolog(log_models=False)

    with mlflow.start_run(log_system_metrics=True):
        gpu_info = os.popen("nvidia-smi").read()
        mlflow.log_text(gpu_info, "gpu-info.txt")

        depthpro = get_depthpro_model(args.patch_size)
        model = LightningModel(depthpro)

        train_loader, val_loader = get_dataloaders(args.batch_size)
        logger = CSVLogger("logs", name=args.run_name)

        checkpoint_callback = ModelCheckpoint(
            monitor='val_psnr',
            mode='max',
            save_top_k=-1,
            save_last=True,
            dirpath="./checkpoints",
            filename='model-{epoch:02d}-{val_psnr:.2f}',
            save_weights_only=True
        )

        early_stopping_callback = EarlyStopping(
            monitor='val_psnr', mode='max', patience=3, verbose=True, min_delta=0.1
        )

        trainer = L.Trainer(
            max_epochs=args.epochs,
            accelerator='gpu',
            devices=1,
            precision='16-mixed',
            logger=logger,
            callbacks=[checkpoint_callback, early_stopping_callback]
        )

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # ✅ 开始训练
        epoch_start = time.time()
        trainer.fit(model, train_loader, val_loader)
        epoch_time = time.time() - epoch_start

        print("Callback metrics:", trainer.callback_metrics)

        # ✅ 记录基础训练信息
        mlflow.log_metrics({
            "epoch_time": epoch_time,
            "trainable_params": trainable_params
        }, step=args.epochs - 1)

        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"
        print("model_uri:", model_uri)

        if checkpoint_callback.best_model_path:
            mlflow.log_artifact(checkpoint_callback.best_model_path, artifact_path="checkpoints")
            print(f"✅ Best model saved at: {checkpoint_callback.best_model_path}")

            # ✅ 保存为精简版 .pth（只包含模型参数）
            torch.save(model.model.state_dict(), "best_model.pth")
            mlflow.log_artifact("best_model.pth", artifact_path="model")
        else:
            print("⚠️ No checkpoint was saved because val_psnr metric was not available.")

        test_results = trainer.test(model, dataloaders=val_loader)
        if test_results:
            mlflow.log_metrics({
                "test_loss": test_results[0].get("test_mse_loss", 0),
                "test_psnr": test_results[0].get("test_psnr", 0),
                "test_ssim": test_results[0].get("test_ssim", 0),
                "test_snr": test_results[0].get("test_snr", 0)
            })

        mlflow.end_run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--experiment_name", type=str, default="SuperResolutionDepthPro")
    parser.add_argument("--run_name", type=str, default="sr-run")
    args = parser.parse_args()
    main(args)
