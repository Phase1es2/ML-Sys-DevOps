### ✅ train.py
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
    #mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://129.114.24.214:8000"))
    mlflow.set_tracking_uri("http://129.114.24.214:8000")
    mlflow.set_experiment("SuperResolutionDepthPro")

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "gamma": args.gamma,
            "patch_size": args.patch_size,
            "model": "DepthProForSuperResolution"
        })

        gpu_info = os.popen("nvidia-smi").read()
        mlflow.log_text(gpu_info, "gpu-info.txt")

        depthpro = get_depthpro_model(args)
        model = LightningModel(depthpro)

        train_loader, val_loader = get_dataloaders(args.batch_size)

        logger = CSVLogger("logs", name=args.run_name)

        checkpoint_callback = ModelCheckpoint(
            monitor='val_psnr', mode='max', save_top_k=1,
            filename='model-{epoch:02d}-{val_psnr:.2f}', save_weights_only=True
        )
        early_stopping_callback = EarlyStopping(
            monitor='val_psnr', mode='max', patience=3, verbose=True, min_delta=0.1
        )
        
        #trainer = L.Trainer(
        #    max_epochs=1, # args.epochs,
        #    accelerator='gpu',
        #    devices=1,
        #    precision=16,
        #    logger=logger,
        #    callbacks=[checkpoint_callback, early_stopping_callback]
        #)

        #trainer.fit(model, train_loader, val_loader)
        
        
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"
        print("model_uri:", model_uri)
        #registered_model = mlflow.register_model(model_uri=model_uri, name="model")
        mlflow.pytorch.log_model(model.model, name="model")
        
        
        mlflow.log_artifact(checkpoint_callback.best_model_path, artifact_path="checkpoints")
        print(f"Best model saved at: {checkpoint_callback.best_model_path}")

        test_results = trainer.test(model, dataloaders=val_loader)
        if test_results:
            mlflow.log_metrics({
                "test_loss": test_results[0]["test_mse_loss"],
                "test_psnr": test_results[0]["test_psnr"],
                "test_ssim": test_results[0]["test_ssim"],
                "test_snr": test_results[0]["test_snr"]
            })

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