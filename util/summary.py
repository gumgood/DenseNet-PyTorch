from torchinfo import summary
from util.config import args
from util.fileio import DataUtil, FilePath
from net import DenseNet


def model_info():
    # Sample one batch
    _, test_dataloader = DataUtil.dataloader(mode='test')
    images, labels = next(iter(test_dataloader))

    # Load model
    model = DenseNet.model()

    # Model summary
    print(FilePath.model_name)
    summary(model=model, device=args.device, input_data=images)


if __name__ == '__main__':
    model_info()
