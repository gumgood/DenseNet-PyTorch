import torch
import torch.nn as nn
from util.config import args


class _DenseLayer(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int, drop_rate: float, bottleneck: bool):
        super().__init__()
        self.layer = nn.Sequential()
        if bottleneck:
            self.layer.append(nn.BatchNorm2d(num_features=in_channels))
            self.layer.append(nn.ReLU(inplace=True))
            self.layer.append(nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, stride=1, padding=0, bias=False))
            if drop_rate > 0.0:
                self.layer.append(nn.Dropout2d(p=drop_rate))
            self.layer.append(nn.BatchNorm2d(num_features=4 * growth_rate))
            self.layer.append(nn.ReLU(inplace=True))
            self.layer.append(nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))
            if drop_rate > 0.0:
                self.layer.append(nn.Dropout2d(p=drop_rate))
        else:
            self.layer.append(nn.BatchNorm2d(num_features=in_channels))
            self.layer.append(nn.ReLU(inplace=True))
            self.layer.append(nn.Conv2d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))
            if drop_rate > 0.0:
                self.layer.append(nn.Dropout2d(p=drop_rate))

    def forward(self, x):
        return self.layer(x)


class _DenseBlock(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int, drop_rate: float, bottleneck: bool, num_layer: int):
        super().__init__()
        self.layers = nn.ModuleList([
            _DenseLayer(in_channels=in_channels + growth_rate * i,
                        growth_rate=growth_rate,
                        drop_rate=drop_rate,
                        bottleneck=bottleneck) for i in range(num_layer)
        ])

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            x = torch.cat(tensors=[x, out], dim=1)
        return x


class _TransitionLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, drop_rate: float):
        super().__init__()
        self.layer = nn.Sequential()
        self.layer.append(nn.BatchNorm2d(num_features=in_channels))
        self.layer.append(nn.ReLU(inplace=True))
        self.layer.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        if drop_rate > 0.0:
            self.layer.append(nn.Dropout2d(p=drop_rate))
        self.layer.append(nn.AvgPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        return self.layer(x)


class DenseNet(nn.Module):
    def __init__(self, num_classes: int, num_layer: int, growth_rate: int, drop_rate: float,
                 bottleneck: bool, compression_factor: float):
        super().__init__()

        assert (num_layer - 4) % 3 == 0
        num_layer = (num_layer - 4) // 3
        if bottleneck:
            num_layer = num_layer // 2

        # First convolution layer
        num_channels = growth_rate * 2 if args.bc else 16
        self.first = nn.Conv2d(3, num_channels, kernel_size=3, stride=1, padding=1, bias=False)

        # Dense block 1 and transition layer
        self.denseblock1 = _DenseBlock(in_channels=num_channels, growth_rate=growth_rate, drop_rate=drop_rate,
                                       bottleneck=bottleneck, num_layer=num_layer)
        num_channels = num_channels + num_layer * growth_rate
        self.transition1 = _TransitionLayer(in_channels=num_channels,
                                            out_channels=int(num_channels * compression_factor), drop_rate=drop_rate)
        num_channels = int(num_channels * compression_factor)

        # Dense block 2 and transition layer
        self.denseblock2 = _DenseBlock(in_channels=num_channels, growth_rate=growth_rate, drop_rate=drop_rate,
                                       bottleneck=bottleneck, num_layer=num_layer)
        num_channels = num_channels + num_layer * growth_rate
        self.transition2 = _TransitionLayer(in_channels=num_channels,
                                            out_channels=int(num_channels * compression_factor), drop_rate=drop_rate)
        num_channels = int(num_channels * compression_factor)

        # Dense block 3 and transition layer
        self.denseblock3 = _DenseBlock(in_channels=num_channels, growth_rate=growth_rate, drop_rate=drop_rate,
                                       bottleneck=bottleneck, num_layer=num_layer)
        num_channels = num_channels + num_layer * growth_rate

        # Last Layer
        self.last = nn.Sequential(
            nn.BatchNorm2d(num_features=num_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(num_channels, num_classes)
        )

        # Weight initialization
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight.data)
            if module.bias is not None:
                nn.init.zeros_(module.bias.data)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight.data)
            if module.bias is not None:
                nn.init.zeros_(module.bias.data)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight.data, mean=0.0, std=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias.data)

    def forward(self, x):
        x = self.first(x)
        x = self.denseblock1(x)
        x = self.transition1(x)
        x = self.denseblock2(x)
        x = self.transition2(x)
        x = self.denseblock3(x)
        x = self.last(x)
        return x


def model():
    num_classes = 10
    num_layer = args.num_layer
    growth_rate = args.growth_rate
    drop_rate = 0.0
    bottleneck = False
    compression_factor = 1.0

    if args.bc:
        bottleneck = True
        compression_factor = 0.5
    if args.dataset in ['cifar100', 'cifar100+']:
        num_classes = 100
    if args.dataset in ['cifar10', 'cifar100', 'svhn']:
        drop_rate = 0.2
    return DenseNet(num_classes, num_layer, growth_rate, drop_rate, bottleneck, compression_factor)


def loss_fn():
    return nn.CrossEntropyLoss()


def optimizer(model):
    return torch.optim.SGD(model.parameters(),
                           lr=args.lr,
                           momentum=args.momentum,
                           nesterov=True,
                           weight_decay=args.weight_decay,
                           )


def lr_scheduler(optimizer):
    milestones = [int(step * args.epochs) for step in [0.5, 0.75]]
    return torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                milestones=milestones,
                                                gamma=0.1
                                                )
