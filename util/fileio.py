import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from util.config import args


class FilePath:
    model_name = "DenseNet_%sL%s_k%s_%s" % ('BC_' if args.bc else '', args.num_layer, args.growth_rate, args.dataset)

    @classmethod
    def data(cls):
        return './datasets'

    @classmethod
    def checkpoint(cls, epoch=args.epochs):
        return f'./models/{cls.model_name}_epoch{epoch}.pth'

    @classmethod
    def tensorboard(cls):
        return f'./runs/{cls.model_name}'


class DataUtil:
    indices = None

    @classmethod
    def dataloader(cls, mode):
        assert mode in ['test', 'train', 'val', 'train+val']

        if args.dataset in ['cifar10']:
            return cls._dataloader_cifar10(mode)
        elif args.dataset in ['cifar10+']:
            return cls._dataloader_cifar10(mode, augment=True)
        elif args.dataset in ['cifar100']:
            return cls._dataloader_cifar100(mode)
        elif args.dataset in ['cifar100+']:
            return cls._dataloader_cifar100(mode, augment=True)
        else:
            return cls._dataloader_svhn(mode)

    @classmethod
    def _dataloader_cifar10(cls, mode, augment=False):
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]

        if mode == 'train' and augment:
            transform = transforms.Compose([
                transforms.Resize(size=32),
                transforms.RandomCrop(size=32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std, inplace=True)
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=32),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std, inplace=True)
            ])

        if mode in ['train', 'val', 'train+val']:
            dataset = datasets.CIFAR10(root=FilePath.data(), train=True, download=True, transform=transform)
        else:
            dataset = datasets.CIFAR10(root=FilePath.data(), train=False, download=True, transform=transform)

        if mode in ['train', 'val'] and cls.indices is None:
            cls.indices = torch.randperm(len(dataset))

        if mode == 'train':
            return DataLoader(Subset(dataset, cls.indices[5000:]), batch_size=args.batch_size)
        elif mode == 'val':
            return DataLoader(Subset(dataset, cls.indices[:5000]), batch_size=args.batch_size)
        elif mode == 'train+val':
            return DataLoader(dataset, batch_size=args.batch_size)
        else:
            return DataLoader(dataset, batch_size=args.batch_size)

    @classmethod
    def _dataloader_cifar100(cls, mode, augment=False):
        mean = [0.5071, 0.4865, 0.4409]
        std = [0.2673, 0.2564, 0.2762]

        if mode == 'train' and augment:
            transform = transforms.Compose([
                transforms.Resize(size=32),
                transforms.RandomCrop(size=32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std, inplace=True)
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=32),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std, inplace=True)
            ])

        if mode in ['train', 'val', 'train+val']:
            dataset = datasets.CIFAR100(root=FilePath.data(), train=True, download=True, transform=transform)
        else:
            dataset = datasets.CIFAR100(root=FilePath.data(), train=False, download=True, transform=transform)

        if mode in ['train', 'val'] and cls.indices is None:
            cls.indices = torch.randperm(len(dataset))

        if mode == 'train':
            return DataLoader(Subset(dataset, cls.indices[5000:]), batch_size=args.batch_size)
        elif mode == 'val':
            return DataLoader(Subset(dataset, cls.indices[:5000]), batch_size=args.batch_size)
        elif mode == 'train+val':
            return DataLoader(dataset, batch_size=args.batch_size)
        else:
            return DataLoader(dataset, batch_size=args.batch_size)

    @classmethod
    def _dataloader_svhn(cls, mode):
        transform = transforms.Compose([
            transforms.Resize(size=32),
            transforms.ToTensor()
        ])

        if mode in ['train', 'val', 'train+val']:
            dataset = datasets.SVHN(root=FilePath.data(), split='train', download=True, transform=transform)
        else:
            dataset = datasets.SVHN(root=FilePath.data(), split='test', download=True, transform=transform)

        if mode in ['train', 'val'] and cls.indices is None:
            cls.indices = torch.randperm(len(dataset))

        if mode == 'train':
            return DataLoader(Subset(dataset, cls.indices[6000:]), batch_size=args.batch_size)
        elif mode == 'val':
            return DataLoader(Subset(dataset, cls.indices[:6000]), batch_size=args.batch_size)
        elif mode == 'train+val':
            return DataLoader(dataset, batch_size=args.batch_size)
        else:
            return DataLoader(dataset, batch_size=args.batch_size)
