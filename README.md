# A PyTorch Implementation for Densely Connected Convolutional Networks

This repository implements [DenseNet (Densely Connected Convolutional Networks)](https://arxiv.org/abs/1608.06993).
Currently, it has DenseNet/DenseNet-BC implementations for CIFAR-10, CIFAR-100, SVHN.

## Requirements

You will need to install the following packages present in the [requirements.txt](./requirements.txt) file.

## Directories

```
├── dataset/          (Datasets)
├── models/           (Checkpoints)
├── runs/             (Tensorboard logs)
│
├── net/
│   └── DenseNet.py
├── util/
│   ├── config.py
│   └── fileio.py
│   └── summary.py
├── test.py
├── train.py
│
├── LICENSE
└── README.md
```

## Usage

The default setting is DenseNet with depth L=40 and growth rate k=12 on CIFAR-10.

As an example, the following command trains a DenseNet with depth L=100 and growth rate k=24 on CIFAR-100:

```
python train.py --num-layer 100 growth-rate 24 --dataset cifar100
```

As another example, the following command trains a DenseNet-BC with depth L=190 and growth rate k=40 on CIFAR-100 with data
augmentation :

```
python train.py --bc --num-layer 190 growth-rate 40 --dataset cifar100+
```

## Result

Error rates (%) on CIFAR and SVHN dataset.

| **Method**  | **Depth** | **Growth Rate** | **C10** | **C10+** | **C100** | **C100+** | **SVHN** |
|-------------|-----------|-----------------|---------|----------|----------|-----------|----------|
| DenseNet    | L=40      | k=12            | 8.45    | 5.63     | -        | -         | -        |
| DenseNet    | L=100     | k=12            | -       | -        | -        | -         | -        |
| DenseNet    | L=100     | k=24            | -       | -        | -        | -         | -        |
| DenseNet-BC | L=100     | k=12            | 7.76    | 4.80     | -        | -         | -        |
| DenseNet-BC | L=250     | k=24            | -       | -        | -        | -         | -        |
| DenseNet-BC | L=190     | k=40            | -       | -        | -        | -         | -        |

## Citation

If you use this work, please cite:

```
@article{Huang2016Densely,
  author  = {Huang, Gao and Liu, Zhuang and Weinberger, Kilian Q.},
  title   = {Densely Connected Convolutional Networks},
  journal = {arXiv preprint arXiv:1608.06993},
  year    = {2016}
}
```