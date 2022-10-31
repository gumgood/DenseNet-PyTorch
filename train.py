import torch
from torch.utils.tensorboard import SummaryWriter

from util.config import args
from util.fileio import DataUtil, FilePath
from net import DenseNet
from test import test


def train(dataloader, model, loss_fn, optimizer, lr_scheduler):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.train()
    tot_loss = 0
    for batch, (images, labels) in enumerate(dataloader):
        images, labels = images.to(args.device), labels.to(args.device)

        # Compute prediction error
        preds = model(images)
        loss = loss_fn(preds, labels)
        tot_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % args.print_freq == 0:
            loss, current = loss.item(), batch * len(images)
            print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')

    lr_scheduler.step()

    tot_loss /= num_batches
    return tot_loss


if __name__ == '__main__':
    # 1. Data loading
    train_dataloader = DataUtil.dataloader(mode='train')
    val_dataloader = DataUtil.dataloader(mode='val')

    # 2  Create model, loss function, optimizer and lr scheduler
    print(f'Model: {FilePath.model_name}')
    model = DenseNet.model().to(args.device)
    loss_fn = DenseNet.loss_fn()
    optimizer = DenseNet.optimizer(model)
    lr_scheduler = DenseNet.lr_scheduler(optimizer)

    # 3. Load checkpoint
    if args.start_epoch:
        path = FilePath.checkpoint(args.start_epoch)
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        print(f'Loaded PyTorch Model State from {path}')

    # 4. Train model
    writer = SummaryWriter(FilePath.tensorboard())
    for epoch in range(args.start_epoch + 1, args.epochs + 1):
        print(f'Epoch {epoch}\n--------------------------------')
        train_loss = train(train_dataloader, model, loss_fn, optimizer, lr_scheduler)
        val_loss, val_acc = test(val_dataloader, model, loss_fn)
        if epoch % args.save_freq == 0 or epoch == args.epochs:
            path = FilePath.checkpoint(epoch=epoch)
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_scheduler_state_dict': lr_scheduler.state_dict()
                        }, path)
            print('Saved PyTorch Model State to', path)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Acc/val', val_acc, epoch)
        writer.flush()
    print('Done!')
    writer.close()
