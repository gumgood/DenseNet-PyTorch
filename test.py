import torch

from util.config import args
from util.fileio import DataUtil, FilePath
from net import DenseNet


def test(dataloader, model, loss_fn, epoch):
    size = len(dataloader.dataset)

    model.eval()
    tot_loss, tot_acc = 0, 0
    print(f'Epoch {epoch}\n--------------------------------')
    with torch.no_grad():
        for batch, (images, labels) in enumerate(dataloader):
            images = images.to(args.device)
            labels = labels.to(args.device)

            # Compute test error
            preds = model(images)
            tot_loss += loss_fn(preds, labels).item()
            tot_acc += (preds.argmax(1) == labels).type(torch.float).sum().item()

            if batch % args.print_freq == 0:
                loss, current = loss.item(), batch * len(images)
                print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')

    tot_loss /= size
    tot_acc /= size

    print(f'Test Error: \n Accuracy: {(100 * tot_acc):>0.1f}%, Avg loss: {tot_loss:>8f} \n')
    return tot_loss, tot_acc


if __name__ == '__main__':
    # 1. Data loading
    dataloader = DataUtil.dataloader(mode='test')

    # 2. Create model, loss function
    print(f'Model: {FilePath.model_name}')
    model = DenseNet.model().to(args.device)
    loss_fn = DenseNet.loss_fn()

    # 3. Load a checkpoint
    path = FilePath.checkpoint(args.epochs)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded PyTorch Model State from {path}')

    # 4. Test model
    test(dataloader, model, loss_fn)
