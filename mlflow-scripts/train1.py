import os
import argparse
import mlflow
import mlflow.pytorch
import torch
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from model import LightningModel, get_depthpro_model
from dataloader import get_dataloaders

def main(args):
    mlflow.set_tracking_uri("http://129.114.24.214:8000")
    mlflow.set_experiment(args.experiment_name)

    model = LightningModel(get_depthpro_model(args.patch_size))
    train_loader, val_loader = get_dataloaders(args.batch_size)

    logger = CSVLogger("logs", name=args.run_name)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_psnr', mode='max', save_top_k=1,
        filename='model-{epoch:02d}-{val_psnr:.2f}', save_weights_only=True
    )
    early_stopping_callback = EarlyStopping(
        monitor='val_psnr', mode='max', patience=3, verbose=True, min_delta=0.1
    )

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision=16,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    # 只在 rank 0 启用 MLflow logging
    if trainer.global_rank == 0:
        mlflow.pytorch.autolog()
        mlflow.start_run(run_name=args.run_name, log_system_metrics=True)

        # 手动 log 非自动追踪参数
        mlflow.log_params({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "gamma": args.gamma,
            "patch_size": args.patch_size,
            "model": "DepthProForSuperResolution"
        })

        # GPU 信息
        try:
            gpu_info = os.popen("nvidia-smi").read()
        except:
            gpu_info = "No GPU info available"
        mlflow.log_text(gpu_info, "gpu-info.txt")

    # 训练
    trainer.fit(model, train_loader, val_loader)

    # 只在主进程记录模型和测试结果
    if trainer.global_rank == 0:
        # 加载 best checkpoint 权重
        best_model_path = checkpoint_callback.best_model_path
        print(f"Best model saved at: {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location="cpu"))
        mlflow.pytorch.log_model(model.model, artifact_path="model")  # 避免注册

        mlflow.log_artifact(best_model_path, artifact_path="checkpoints")

        # 测试
        test_results = trainer.test(model, dataloaders=val_loader)
        if test_results:
            mlflow.log_metrics({
                "test_loss": test_results[0].get("test_mse_loss", 0.0),
                "test_psnr": test_results[0].get("test_psnr", 0.0),
                "test_ssim": test_results[0].get("test_ssim", 0.0),
                "test_snr": test_results[0].get("test_snr", 0.0),
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
