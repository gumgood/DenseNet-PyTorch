import torch
from torch.utils.tensorboard import SummaryWriter

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
    # 1. Data loading
    train_dataloader = DataUtil.dataloader(mode='train+val')
    val_dataloader = DataUtil.dataloader(mode='test')

    # 2  Create model, loss function, optimizer and lr scheduler
    print(f'Model: {FilePath.model_name}')
    model = DenseNet.model().to(args.device)
    loss_fn = DenseNet.loss_fn().to(args.device)
    optimizer = DenseNet.optimizer(model)
    lr_scheduler = DenseNet.lr_scheduler(optimizer)

    # 3. Load checkpoint
    if args.start_epoch:
        path = FilePath.checkpoint(args.start_epoch)
        checkpoint = torch.load(path)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        print(f'Loaded PyTorch Model State from {path}')
    else:
        args.start_epoch = 1

    # 4. Train model
    writer = SummaryWriter(FilePath.tensorboard())
    for epoch in range(args.start_epoch, args.epochs + 1):
        train_loss = train(train_dataloader, model, loss_fn, optimizer, epoch)
        val_loss, val_acc = test(val_dataloader, model, loss_fn, epoch)
        lr_scheduler.step()

        if epoch % args.save_freq == 0 or epoch == args.epochs:
            path = FilePath.checkpoint(epoch=epoch)
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
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
