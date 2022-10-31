import torch

from util.config import args
from util.fileio import DataUtil, FilePath
from net import DenseNet


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()
    tot_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(args.device), labels.to(args.device)

            # Compute test error
            preds = model(images)
            tot_loss += loss_fn(preds, labels).item()
            correct += (preds.argmax(1) == labels).type(torch.float).sum().item()
    tot_loss /= num_batches
    correct /= size

    print(f'Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {tot_loss:>8f} \n')
    return tot_loss, correct


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
