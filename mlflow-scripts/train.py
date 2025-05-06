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
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "gamma": args.gamma,
            "patch_size": args.patch_size
        })

        # Load pretrained model
        depthpro = get_depthpro_model(args)
        model = LightningModel(depthpro)

        # Get dataloaders
        train_loader, val_loader = get_dataloaders(args.batch_size)

        # Logger & Callbacks
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
            accelerator='gpu',
            devices=1,
            precision=16,
            logger=logger,
            callbacks=[checkpoint_callback, early_stopping_callback]
        )

        trainer.fit(model, train_loader, val_loader)

        # Save model to MLflow
        mlflow.pytorch.log_model(model.model, artifact_path="model")
        mlflow.log_artifact(checkpoint_callback.best_model_path, artifact_path="checkpoints")
        print(f"Best model saved at: {checkpoint_callback.best_model_path}")

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

