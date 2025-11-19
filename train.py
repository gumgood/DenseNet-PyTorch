import torch
import mlflow
import mlflow.pytorch
import os

from util.config import args
from util.fileio import DataUtil, FilePath
from net import DenseNet
from test import test


def train(dataloader, model, loss_fn, optimizer, epoch):
    size = len(dataloader.dataset)

    model.train()
    tot_loss = 0
    print(f'Epoch {epoch}\n--------------------------------')
    for batch, (images, labels) in enumerate(dataloader):
        images = images.to(args.device)
        labels = labels.to(args.device)

        # Compute prediction error
        preds = model(images)
        loss = loss_fn(preds, labels)

        tot_loss += loss.item() * images.size(0)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % args.print_freq == 0:
            loss, current = loss.item(), batch * len(images)
            print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')

    tot_loss /= size
    return tot_loss


if __name__ == '__main__':
    # Set MLflow experiment
    mlflow.set_experiment(FilePath.model_name)

    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params(vars(args))

        # 1. Data loading
        train_dataloader = DataUtil.dataloader(mode='train+val')
        val_dataloader = DataUtil.dataloader(mode='test')

        # 2. Create model, loss function, optimizer and lr scheduler
        print(f'Model: {FilePath.model_name}')
        model = DenseNet.model().to(args.device)
        loss_fn = DenseNet.loss_fn().to(args.device)
        optimizer = DenseNet.optimizer(model)
        lr_scheduler = DenseNet.lr_scheduler(optimizer)

        # 3. Resume from a previous run if specified
        if args.resume_from_run_id:
            print(f"Resuming training from MLflow run: {args.resume_from_run_id}")
            run_uri = f"runs:/{args.resume_from_run_id}"

            # Restore model
            model = mlflow.pytorch.load_model(f"{run_uri}/model")

            # Restore optimizer
            optimizer_path = mlflow.artifacts.download_artifacts(artifact_uri=f"{run_uri}/checkpoints/optimizer.pth")
            optimizer.load_state_dict(torch.load(optimizer_path))
            os.remove(optimizer_path)


            # Restore scheduler
            scheduler_path = mlflow.artifacts.download_artifacts(artifact_uri=f"{run_uri}/checkpoints/lr_scheduler.pth")
            lr_scheduler.load_state_dict(torch.load(scheduler_path))
            os.remove(scheduler_path)

        # 4. Train model
        for epoch in range(1, args.epochs + 1):
            train_loss = train(train_dataloader, model, loss_fn, optimizer, epoch)
            val_loss, val_acc = test(val_dataloader, model, loss_fn, epoch)
            lr_scheduler.step()

            # Log metrics
            mlflow.log_metric('Loss/train', train_loss, step=epoch)
            mlflow.log_metric('Loss/val', val_loss, step=epoch)
            mlflow.log_metric('Acc/val', val_acc, step=epoch)

        # Log the trained model
        mlflow.pytorch.log_model(model, "model")

        # Save and log optimizer and scheduler states
        torch.save(optimizer.state_dict(), "optimizer.pth")
        mlflow.log_artifact("optimizer.pth", "checkpoints")
        os.remove("optimizer.pth")

        torch.save(lr_scheduler.state_dict(), "lr_scheduler.pth")
        mlflow.log_artifact("lr_scheduler.pth", "checkpoints")
        os.remove("lr_scheduler.pth")


        print('Done!')
