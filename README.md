# Cross-feature Contrastive Loss
This repository implements the momentum tracking algorithm proposed in "Momentum Tracking: Momentum Acceleration for Decentralized Deep Learning on Heterogeneous Data"


# Available Models
* ResNet
* VGG-11
* MobileNet-V2
* LeNet-5

# Available Datasets
* CIFAR-10
* CIFAR-100
* Fashion MNIST
* Imagenette
* Imagenet

# Available Graph Topologies
* Ring Graph
* Torus Graph
* Fully Connected Graph

# Requirements
* found in environment.yml file

# Hyper-parameters
* --world_size  = total number of agents
* --graph       = graph topology (default ring); options: [ring, torus, full]
* --neighbors   = number of neighbor per agent (default 2)
* --arch        = model to train
* --normtype    = type of normalization layer
* --dataset     = dataset to train; ; options: [cifar10, cifar100, fmnist, imagenette, imagenet]
* --batch_size  = batch size for training (batch_size = batch_size per agent x world_size)
* --epochs      = total number of training epochs
* --lr          = learning rate
* --momentum    = momentum coefficient
* --gamma       = averaging rate for gossip 
* --skew        = amount of skew in the data distribution; 1.0 = completely non-IID and 0 = random sampling (IID)


# How to run?


ResNet-20 with 10 agents ring topology:
```
python trainer.py --lr=0.1  --batch-size=320  --world_size=10 --skew=1 --normtype=evonorm --epochs=200 --arch=resnet --momentum=0.9 --graph=ring --neighbors=2 --depth=20 --dataset=cifar10 --classes=10 --devices=4 --seed=123

```

