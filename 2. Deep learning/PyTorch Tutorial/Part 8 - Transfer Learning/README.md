
# Transfer Learning

In this notebook, you'll learn how to use pre-trained networks to solved challenging problems in computer vision. Specifically, you'll use networks trained on [ImageNet](http://www.image-net.org/) [available from torchvision](http://pytorch.org/docs/0.3.0/torchvision/models.html). 

ImageNet is a massive dataset with over 1 million labeled images in 1000 categories. It's used to train deep neural networks using an architecture called convolutional layers. I'm not going to get into the details of convolutional networks here, but if you want to learn more about them, please [watch this](https://www.youtube.com/watch?v=2-Ol7ZB0MmU).

Once trained, these models work astonishingly well as feature detectors for images they weren't trained on. Using a pre-trained network on images not in the training set is called transfer learning. Here we'll use transfer learning to train a network that can classify our cat and dog photos with near perfect accuracy.

With `torchvision.models` you can download these pre-trained networks and use them in your applications. We'll include `models` in our imports now.


```python
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
```

Most of the pretrained models require the input to be 224x224 images. Also, we'll need to match the normalization used when the models were trained. Each color channel was normalized separately, the means are `[0.485, 0.456, 0.406]` and the standard deviations are `[0.229, 0.224, 0.225]`.


```python
data_dir = 'Cat_Dog_data'

# TODO: Define transforms for the training data and testing data
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
```

We can load in a model such as [DenseNet](http://pytorch.org/docs/0.3.0/torchvision/models.html#id5). Let's print out the model architecture so we can see what's going on.


```python
model = models.densenet121(pretrained=True)
model
```

    /opt/conda/lib/python3.6/site-packages/torchvision-0.2.1-py3.6.egg/torchvision/models/densenet.py:212: UserWarning: nn.init.kaiming_normal is now deprecated in favor of nn.init.kaiming_normal_.
      nn.init.kaiming_normal(m.weight.data)





    DenseNet(
      (features): Sequential(
        (conv0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        (norm0): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu0): ReLU(inplace)
        (pool0): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        (denseblock1): _DenseBlock(
          (denselayer1): _DenseLayer(
            (norm1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer2): _DenseLayer(
            (norm1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(96, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer3): _DenseLayer(
            (norm1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer4): _DenseLayer(
            (norm1): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(160, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer5): _DenseLayer(
            (norm1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(192, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer6): _DenseLayer(
            (norm1): BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(224, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
        )
        (transition1): _Transition(
          (norm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (denseblock2): _DenseBlock(
          (denselayer1): _DenseLayer(
            (norm1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer2): _DenseLayer(
            (norm1): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(160, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer3): _DenseLayer(
            (norm1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(192, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer4): _DenseLayer(
            (norm1): BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(224, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer5): _DenseLayer(
            (norm1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer6): _DenseLayer(
            (norm1): BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(288, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer7): _DenseLayer(
            (norm1): BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(320, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer8): _DenseLayer(
            (norm1): BatchNorm2d(352, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(352, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer9): _DenseLayer(
            (norm1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(384, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer10): _DenseLayer(
            (norm1): BatchNorm2d(416, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(416, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer11): _DenseLayer(
            (norm1): BatchNorm2d(448, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(448, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer12): _DenseLayer(
            (norm1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(480, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
        )
        (transition2): _Transition(
          (norm): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (denseblock3): _DenseBlock(
          (denselayer1): _DenseLayer(
            (norm1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer2): _DenseLayer(
            (norm1): BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(288, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer3): _DenseLayer(
            (norm1): BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(320, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer4): _DenseLayer(
            (norm1): BatchNorm2d(352, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(352, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer5): _DenseLayer(
            (norm1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(384, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer6): _DenseLayer(
            (norm1): BatchNorm2d(416, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(416, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer7): _DenseLayer(
            (norm1): BatchNorm2d(448, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(448, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer8): _DenseLayer(
            (norm1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(480, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer9): _DenseLayer(
            (norm1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer10): _DenseLayer(
            (norm1): BatchNorm2d(544, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(544, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer11): _DenseLayer(
            (norm1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(576, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer12): _DenseLayer(
            (norm1): BatchNorm2d(608, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(608, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer13): _DenseLayer(
            (norm1): BatchNorm2d(640, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(640, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer14): _DenseLayer(
            (norm1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(672, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer15): _DenseLayer(
            (norm1): BatchNorm2d(704, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(704, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer16): _DenseLayer(
            (norm1): BatchNorm2d(736, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(736, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer17): _DenseLayer(
            (norm1): BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer18): _DenseLayer(
            (norm1): BatchNorm2d(800, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(800, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer19): _DenseLayer(
            (norm1): BatchNorm2d(832, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(832, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer20): _DenseLayer(
            (norm1): BatchNorm2d(864, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(864, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer21): _DenseLayer(
            (norm1): BatchNorm2d(896, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(896, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer22): _DenseLayer(
            (norm1): BatchNorm2d(928, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(928, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer23): _DenseLayer(
            (norm1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(960, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer24): _DenseLayer(
            (norm1): BatchNorm2d(992, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(992, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
        )
        (transition3): _Transition(
          (norm): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (denseblock4): _DenseBlock(
          (denselayer1): _DenseLayer(
            (norm1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer2): _DenseLayer(
            (norm1): BatchNorm2d(544, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(544, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer3): _DenseLayer(
            (norm1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(576, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer4): _DenseLayer(
            (norm1): BatchNorm2d(608, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(608, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer5): _DenseLayer(
            (norm1): BatchNorm2d(640, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(640, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer6): _DenseLayer(
            (norm1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(672, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer7): _DenseLayer(
            (norm1): BatchNorm2d(704, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(704, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer8): _DenseLayer(
            (norm1): BatchNorm2d(736, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(736, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer9): _DenseLayer(
            (norm1): BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer10): _DenseLayer(
            (norm1): BatchNorm2d(800, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(800, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer11): _DenseLayer(
            (norm1): BatchNorm2d(832, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(832, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer12): _DenseLayer(
            (norm1): BatchNorm2d(864, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(864, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer13): _DenseLayer(
            (norm1): BatchNorm2d(896, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(896, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer14): _DenseLayer(
            (norm1): BatchNorm2d(928, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(928, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer15): _DenseLayer(
            (norm1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(960, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer16): _DenseLayer(
            (norm1): BatchNorm2d(992, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(992, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
        )
        (norm5): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (classifier): Linear(in_features=1024, out_features=1000, bias=True)
    )



This model is built out of two main parts, the features and the classifier. The features part is a stack of convolutional layers and overall works as a feature detector that can be fed into a classifier. The classifier part is a single fully-connected layer `(classifier): Linear(in_features=1024, out_features=1000)`. This layer was trained on the ImageNet dataset, so it won't work for our specific problem. That means we need to replace the classifier, but the features will work perfectly on their own. In general, I think about pre-trained networks as amazingly good feature detectors that can be used as the input for simple feed-forward classifiers.


```python
# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, 500)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(500, 2)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier
```

With our model built, we need to train the classifier. However, now we're using a **really deep** neural network. If you try to train this on a CPU like normal, it will take a long, long time. Instead, we're going to use the GPU to do the calculations. The linear algebra computations are done in parallel on the GPU leading to 100x increased training speeds. It's also possible to train on multiple GPUs, further decreasing training time.

PyTorch, along with pretty much every other deep learning framework, uses [CUDA](https://developer.nvidia.com/cuda-zone) to efficiently compute the forward and backwards passes on the GPU. In PyTorch, you move your model parameters and other tensors to the GPU memory using `model.to('cuda')`. You can move them back from the GPU with `model.to('cpu')` which you'll commonly do when you need to operate on the network output outside of PyTorch. As a demonstration of the increased speed, I'll compare how long it takes to perform a forward and backward pass with and without a GPU.


```python
import time
```


```python
# for device in ['cpu', 'cuda']:

#     criterion = nn.NLLLoss()
#     # Only train the classifier parameters, feature parameters are frozen
#     optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

#     model.to(device)

#     for ii, (inputs, labels) in enumerate(trainloader):
#         print(inputs.size())
#         print(labels.size())
#         # Move input and label tensors to the GPU
#         inputs, labels = inputs.to(device), labels.to(device)

#         start = time.time()

#         outputs = model.forward(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         if ii==3:
#             break
        
#     print(f"Device = {device}; Time per batch: {(time.time() - start)/3:.3f} seconds")
```

    torch.Size([64, 3, 224, 224])
    torch.Size([64])
    torch.Size([64, 3, 224, 224])
    torch.Size([64])
    torch.Size([64, 3, 224, 224])
    torch.Size([64])
    torch.Size([64, 3, 224, 224])
    torch.Size([64])
    Device = cpu; Time per batch: 7.714 seconds
    torch.Size([64, 3, 224, 224])
    torch.Size([64])
    torch.Size([64, 3, 224, 224])
    torch.Size([64])
    torch.Size([64, 3, 224, 224])
    torch.Size([64])
    torch.Size([64, 3, 224, 224])
    torch.Size([64])
    Device = cuda; Time per batch: 0.010 seconds


You can write device agnostic code which will automatically use CUDA if it's enabled like so:
```python
# at beginning of the script
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

...

# then whenever you get a new Tensor or Module
# this won't copy if they are already on the desired device
input = data.to(device)
model = MyModule(...).to(device)
```

From here, I'll let you finish training the model. The process is the same as before except now your model is much more powerful. You should get better than 95% accuracy easily.

>**Exercise:** Train a pretrained models to classify the cat and dog images. Continue with the DenseNet model, or try ResNet, it's also a good model to try out first. Make sure you are only training the classifier and the parameters for the features part are frozen.


```python
criterion = nn.NLLLoss()
# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

device = 'cuda'
model.to(device)
```




    DenseNet(
      (features): Sequential(
        (conv0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        (norm0): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu0): ReLU(inplace)
        (pool0): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        (denseblock1): _DenseBlock(
          (denselayer1): _DenseLayer(
            (norm1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer2): _DenseLayer(
            (norm1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(96, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer3): _DenseLayer(
            (norm1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer4): _DenseLayer(
            (norm1): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(160, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer5): _DenseLayer(
            (norm1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(192, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer6): _DenseLayer(
            (norm1): BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(224, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
        )
        (transition1): _Transition(
          (norm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (denseblock2): _DenseBlock(
          (denselayer1): _DenseLayer(
            (norm1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer2): _DenseLayer(
            (norm1): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(160, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer3): _DenseLayer(
            (norm1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(192, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer4): _DenseLayer(
            (norm1): BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(224, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer5): _DenseLayer(
            (norm1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer6): _DenseLayer(
            (norm1): BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(288, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer7): _DenseLayer(
            (norm1): BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(320, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer8): _DenseLayer(
            (norm1): BatchNorm2d(352, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(352, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer9): _DenseLayer(
            (norm1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(384, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer10): _DenseLayer(
            (norm1): BatchNorm2d(416, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(416, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer11): _DenseLayer(
            (norm1): BatchNorm2d(448, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(448, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer12): _DenseLayer(
            (norm1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(480, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
        )
        (transition2): _Transition(
          (norm): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (denseblock3): _DenseBlock(
          (denselayer1): _DenseLayer(
            (norm1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer2): _DenseLayer(
            (norm1): BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(288, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer3): _DenseLayer(
            (norm1): BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(320, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer4): _DenseLayer(
            (norm1): BatchNorm2d(352, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(352, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer5): _DenseLayer(
            (norm1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(384, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer6): _DenseLayer(
            (norm1): BatchNorm2d(416, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(416, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer7): _DenseLayer(
            (norm1): BatchNorm2d(448, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(448, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer8): _DenseLayer(
            (norm1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(480, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer9): _DenseLayer(
            (norm1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer10): _DenseLayer(
            (norm1): BatchNorm2d(544, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(544, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer11): _DenseLayer(
            (norm1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(576, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer12): _DenseLayer(
            (norm1): BatchNorm2d(608, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(608, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer13): _DenseLayer(
            (norm1): BatchNorm2d(640, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(640, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer14): _DenseLayer(
            (norm1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(672, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer15): _DenseLayer(
            (norm1): BatchNorm2d(704, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(704, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer16): _DenseLayer(
            (norm1): BatchNorm2d(736, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(736, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer17): _DenseLayer(
            (norm1): BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer18): _DenseLayer(
            (norm1): BatchNorm2d(800, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(800, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer19): _DenseLayer(
            (norm1): BatchNorm2d(832, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(832, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer20): _DenseLayer(
            (norm1): BatchNorm2d(864, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(864, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer21): _DenseLayer(
            (norm1): BatchNorm2d(896, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(896, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer22): _DenseLayer(
            (norm1): BatchNorm2d(928, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(928, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer23): _DenseLayer(
            (norm1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(960, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer24): _DenseLayer(
            (norm1): BatchNorm2d(992, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(992, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
        )
        (transition3): _Transition(
          (norm): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (denseblock4): _DenseBlock(
          (denselayer1): _DenseLayer(
            (norm1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer2): _DenseLayer(
            (norm1): BatchNorm2d(544, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(544, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer3): _DenseLayer(
            (norm1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(576, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer4): _DenseLayer(
            (norm1): BatchNorm2d(608, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(608, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer5): _DenseLayer(
            (norm1): BatchNorm2d(640, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(640, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer6): _DenseLayer(
            (norm1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(672, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer7): _DenseLayer(
            (norm1): BatchNorm2d(704, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(704, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer8): _DenseLayer(
            (norm1): BatchNorm2d(736, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(736, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer9): _DenseLayer(
            (norm1): BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer10): _DenseLayer(
            (norm1): BatchNorm2d(800, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(800, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer11): _DenseLayer(
            (norm1): BatchNorm2d(832, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(832, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer12): _DenseLayer(
            (norm1): BatchNorm2d(864, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(864, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer13): _DenseLayer(
            (norm1): BatchNorm2d(896, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(896, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer14): _DenseLayer(
            (norm1): BatchNorm2d(928, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(928, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer15): _DenseLayer(
            (norm1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(960, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer16): _DenseLayer(
            (norm1): BatchNorm2d(992, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(992, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
        )
        (norm5): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (classifier): Sequential(
        (fc1): Linear(in_features=1024, out_features=500, bias=True)
        (relu): ReLU()
        (fc2): Linear(in_features=500, out_features=2, bias=True)
        (output): LogSoftmax()
      )
    )




```python
# TODO: Train a model with a pre-trained network
def validation(model, testloader, criterion):
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:

#         images.resize_(images.shape[0], 224*224)
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy
```


```python
epochs = 2
steps = 0
running_loss = 0
print_every = 40
for e in range(epochs):
    model.train()
    for images, labels in trainloader:
        steps += 1
        
        # Flatten images into a 224*224 long vector
#         images.resize_(images.size()[0], 224*224)
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            # Make sure network is in eval mode for inference
            model.eval()
            
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                test_loss, accuracy = validation(model, testloader, criterion)
                
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                  "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
            
            running_loss = 0
            
            # Make sure training is back on
            model.train()
```

    Epoch: 1/2..  Training Loss: 0.197..  Test Loss: 0.046..  Test Accuracy: 0.983
    Epoch: 1/2..  Training Loss: 0.165..  Test Loss: 0.044..  Test Accuracy: 0.983
    Epoch: 1/2..  Training Loss: 0.185..  Test Loss: 0.073..  Test Accuracy: 0.969
    Epoch: 1/2..  Training Loss: 0.168..  Test Loss: 0.043..  Test Accuracy: 0.985
    Epoch: 1/2..  Training Loss: 0.151..  Test Loss: 0.041..  Test Accuracy: 0.984
    Epoch: 1/2..  Training Loss: 0.146..  Test Loss: 0.053..  Test Accuracy: 0.981
    Epoch: 1/2..  Training Loss: 0.150..  Test Loss: 0.081..  Test Accuracy: 0.971
    Epoch: 1/2..  Training Loss: 0.180..  Test Loss: 0.060..  Test Accuracy: 0.978
    Epoch: 2/2..  Training Loss: 0.141..  Test Loss: 0.037..  Test Accuracy: 0.985
    Epoch: 2/2..  Training Loss: 0.154..  Test Loss: 0.038..  Test Accuracy: 0.985
    Epoch: 2/2..  Training Loss: 0.135..  Test Loss: 0.055..  Test Accuracy: 0.979
    Epoch: 2/2..  Training Loss: 0.156..  Test Loss: 0.036..  Test Accuracy: 0.985
    Epoch: 2/2..  Training Loss: 0.165..  Test Loss: 0.035..  Test Accuracy: 0.987
    Epoch: 2/2..  Training Loss: 0.142..  Test Loss: 0.045..  Test Accuracy: 0.983
    Epoch: 2/2..  Training Loss: 0.157..  Test Loss: 0.037..  Test Accuracy: 0.986
    Epoch: 2/2..  Training Loss: 0.148..  Test Loss: 0.035..  Test Accuracy: 0.985
    Epoch: 2/2..  Training Loss: 0.146..  Test Loss: 0.046..  Test Accuracy: 0.981



```python
# checkpoint = {'input_size': 1024,
#               'output_size': 2,
#               'hidden_layers': [each.out_features for each in model.hidden_layers],
#               'state_dict': model.state_dict()}

torch.save(model.state_dict(), 'checkpoint.pth')

```


```python
state_dict = torch.load('checkpoint.pth')
print(state_dict.keys())
```

    odict_keys(['features.conv0.weight', 'features.norm0.weight', 'features.norm0.bias', 'features.norm0.running_mean', 'features.norm0.running_var', 'features.denseblock1.denselayer1.norm1.weight', 'features.denseblock1.denselayer1.norm1.bias', 'features.denseblock1.denselayer1.norm1.running_mean', 'features.denseblock1.denselayer1.norm1.running_var', 'features.denseblock1.denselayer1.conv1.weight', 'features.denseblock1.denselayer1.norm2.weight', 'features.denseblock1.denselayer1.norm2.bias', 'features.denseblock1.denselayer1.norm2.running_mean', 'features.denseblock1.denselayer1.norm2.running_var', 'features.denseblock1.denselayer1.conv2.weight', 'features.denseblock1.denselayer2.norm1.weight', 'features.denseblock1.denselayer2.norm1.bias', 'features.denseblock1.denselayer2.norm1.running_mean', 'features.denseblock1.denselayer2.norm1.running_var', 'features.denseblock1.denselayer2.conv1.weight', 'features.denseblock1.denselayer2.norm2.weight', 'features.denseblock1.denselayer2.norm2.bias', 'features.denseblock1.denselayer2.norm2.running_mean', 'features.denseblock1.denselayer2.norm2.running_var', 'features.denseblock1.denselayer2.conv2.weight', 'features.denseblock1.denselayer3.norm1.weight', 'features.denseblock1.denselayer3.norm1.bias', 'features.denseblock1.denselayer3.norm1.running_mean', 'features.denseblock1.denselayer3.norm1.running_var', 'features.denseblock1.denselayer3.conv1.weight', 'features.denseblock1.denselayer3.norm2.weight', 'features.denseblock1.denselayer3.norm2.bias', 'features.denseblock1.denselayer3.norm2.running_mean', 'features.denseblock1.denselayer3.norm2.running_var', 'features.denseblock1.denselayer3.conv2.weight', 'features.denseblock1.denselayer4.norm1.weight', 'features.denseblock1.denselayer4.norm1.bias', 'features.denseblock1.denselayer4.norm1.running_mean', 'features.denseblock1.denselayer4.norm1.running_var', 'features.denseblock1.denselayer4.conv1.weight', 'features.denseblock1.denselayer4.norm2.weight', 'features.denseblock1.denselayer4.norm2.bias', 'features.denseblock1.denselayer4.norm2.running_mean', 'features.denseblock1.denselayer4.norm2.running_var', 'features.denseblock1.denselayer4.conv2.weight', 'features.denseblock1.denselayer5.norm1.weight', 'features.denseblock1.denselayer5.norm1.bias', 'features.denseblock1.denselayer5.norm1.running_mean', 'features.denseblock1.denselayer5.norm1.running_var', 'features.denseblock1.denselayer5.conv1.weight', 'features.denseblock1.denselayer5.norm2.weight', 'features.denseblock1.denselayer5.norm2.bias', 'features.denseblock1.denselayer5.norm2.running_mean', 'features.denseblock1.denselayer5.norm2.running_var', 'features.denseblock1.denselayer5.conv2.weight', 'features.denseblock1.denselayer6.norm1.weight', 'features.denseblock1.denselayer6.norm1.bias', 'features.denseblock1.denselayer6.norm1.running_mean', 'features.denseblock1.denselayer6.norm1.running_var', 'features.denseblock1.denselayer6.conv1.weight', 'features.denseblock1.denselayer6.norm2.weight', 'features.denseblock1.denselayer6.norm2.bias', 'features.denseblock1.denselayer6.norm2.running_mean', 'features.denseblock1.denselayer6.norm2.running_var', 'features.denseblock1.denselayer6.conv2.weight', 'features.transition1.norm.weight', 'features.transition1.norm.bias', 'features.transition1.norm.running_mean', 'features.transition1.norm.running_var', 'features.transition1.conv.weight', 'features.denseblock2.denselayer1.norm1.weight', 'features.denseblock2.denselayer1.norm1.bias', 'features.denseblock2.denselayer1.norm1.running_mean', 'features.denseblock2.denselayer1.norm1.running_var', 'features.denseblock2.denselayer1.conv1.weight', 'features.denseblock2.denselayer1.norm2.weight', 'features.denseblock2.denselayer1.norm2.bias', 'features.denseblock2.denselayer1.norm2.running_mean', 'features.denseblock2.denselayer1.norm2.running_var', 'features.denseblock2.denselayer1.conv2.weight', 'features.denseblock2.denselayer2.norm1.weight', 'features.denseblock2.denselayer2.norm1.bias', 'features.denseblock2.denselayer2.norm1.running_mean', 'features.denseblock2.denselayer2.norm1.running_var', 'features.denseblock2.denselayer2.conv1.weight', 'features.denseblock2.denselayer2.norm2.weight', 'features.denseblock2.denselayer2.norm2.bias', 'features.denseblock2.denselayer2.norm2.running_mean', 'features.denseblock2.denselayer2.norm2.running_var', 'features.denseblock2.denselayer2.conv2.weight', 'features.denseblock2.denselayer3.norm1.weight', 'features.denseblock2.denselayer3.norm1.bias', 'features.denseblock2.denselayer3.norm1.running_mean', 'features.denseblock2.denselayer3.norm1.running_var', 'features.denseblock2.denselayer3.conv1.weight', 'features.denseblock2.denselayer3.norm2.weight', 'features.denseblock2.denselayer3.norm2.bias', 'features.denseblock2.denselayer3.norm2.running_mean', 'features.denseblock2.denselayer3.norm2.running_var', 'features.denseblock2.denselayer3.conv2.weight', 'features.denseblock2.denselayer4.norm1.weight', 'features.denseblock2.denselayer4.norm1.bias', 'features.denseblock2.denselayer4.norm1.running_mean', 'features.denseblock2.denselayer4.norm1.running_var', 'features.denseblock2.denselayer4.conv1.weight', 'features.denseblock2.denselayer4.norm2.weight', 'features.denseblock2.denselayer4.norm2.bias', 'features.denseblock2.denselayer4.norm2.running_mean', 'features.denseblock2.denselayer4.norm2.running_var', 'features.denseblock2.denselayer4.conv2.weight', 'features.denseblock2.denselayer5.norm1.weight', 'features.denseblock2.denselayer5.norm1.bias', 'features.denseblock2.denselayer5.norm1.running_mean', 'features.denseblock2.denselayer5.norm1.running_var', 'features.denseblock2.denselayer5.conv1.weight', 'features.denseblock2.denselayer5.norm2.weight', 'features.denseblock2.denselayer5.norm2.bias', 'features.denseblock2.denselayer5.norm2.running_mean', 'features.denseblock2.denselayer5.norm2.running_var', 'features.denseblock2.denselayer5.conv2.weight', 'features.denseblock2.denselayer6.norm1.weight', 'features.denseblock2.denselayer6.norm1.bias', 'features.denseblock2.denselayer6.norm1.running_mean', 'features.denseblock2.denselayer6.norm1.running_var', 'features.denseblock2.denselayer6.conv1.weight', 'features.denseblock2.denselayer6.norm2.weight', 'features.denseblock2.denselayer6.norm2.bias', 'features.denseblock2.denselayer6.norm2.running_mean', 'features.denseblock2.denselayer6.norm2.running_var', 'features.denseblock2.denselayer6.conv2.weight', 'features.denseblock2.denselayer7.norm1.weight', 'features.denseblock2.denselayer7.norm1.bias', 'features.denseblock2.denselayer7.norm1.running_mean', 'features.denseblock2.denselayer7.norm1.running_var', 'features.denseblock2.denselayer7.conv1.weight', 'features.denseblock2.denselayer7.norm2.weight', 'features.denseblock2.denselayer7.norm2.bias', 'features.denseblock2.denselayer7.norm2.running_mean', 'features.denseblock2.denselayer7.norm2.running_var', 'features.denseblock2.denselayer7.conv2.weight', 'features.denseblock2.denselayer8.norm1.weight', 'features.denseblock2.denselayer8.norm1.bias', 'features.denseblock2.denselayer8.norm1.running_mean', 'features.denseblock2.denselayer8.norm1.running_var', 'features.denseblock2.denselayer8.conv1.weight', 'features.denseblock2.denselayer8.norm2.weight', 'features.denseblock2.denselayer8.norm2.bias', 'features.denseblock2.denselayer8.norm2.running_mean', 'features.denseblock2.denselayer8.norm2.running_var', 'features.denseblock2.denselayer8.conv2.weight', 'features.denseblock2.denselayer9.norm1.weight', 'features.denseblock2.denselayer9.norm1.bias', 'features.denseblock2.denselayer9.norm1.running_mean', 'features.denseblock2.denselayer9.norm1.running_var', 'features.denseblock2.denselayer9.conv1.weight', 'features.denseblock2.denselayer9.norm2.weight', 'features.denseblock2.denselayer9.norm2.bias', 'features.denseblock2.denselayer9.norm2.running_mean', 'features.denseblock2.denselayer9.norm2.running_var', 'features.denseblock2.denselayer9.conv2.weight', 'features.denseblock2.denselayer10.norm1.weight', 'features.denseblock2.denselayer10.norm1.bias', 'features.denseblock2.denselayer10.norm1.running_mean', 'features.denseblock2.denselayer10.norm1.running_var', 'features.denseblock2.denselayer10.conv1.weight', 'features.denseblock2.denselayer10.norm2.weight', 'features.denseblock2.denselayer10.norm2.bias', 'features.denseblock2.denselayer10.norm2.running_mean', 'features.denseblock2.denselayer10.norm2.running_var', 'features.denseblock2.denselayer10.conv2.weight', 'features.denseblock2.denselayer11.norm1.weight', 'features.denseblock2.denselayer11.norm1.bias', 'features.denseblock2.denselayer11.norm1.running_mean', 'features.denseblock2.denselayer11.norm1.running_var', 'features.denseblock2.denselayer11.conv1.weight', 'features.denseblock2.denselayer11.norm2.weight', 'features.denseblock2.denselayer11.norm2.bias', 'features.denseblock2.denselayer11.norm2.running_mean', 'features.denseblock2.denselayer11.norm2.running_var', 'features.denseblock2.denselayer11.conv2.weight', 'features.denseblock2.denselayer12.norm1.weight', 'features.denseblock2.denselayer12.norm1.bias', 'features.denseblock2.denselayer12.norm1.running_mean', 'features.denseblock2.denselayer12.norm1.running_var', 'features.denseblock2.denselayer12.conv1.weight', 'features.denseblock2.denselayer12.norm2.weight', 'features.denseblock2.denselayer12.norm2.bias', 'features.denseblock2.denselayer12.norm2.running_mean', 'features.denseblock2.denselayer12.norm2.running_var', 'features.denseblock2.denselayer12.conv2.weight', 'features.transition2.norm.weight', 'features.transition2.norm.bias', 'features.transition2.norm.running_mean', 'features.transition2.norm.running_var', 'features.transition2.conv.weight', 'features.denseblock3.denselayer1.norm1.weight', 'features.denseblock3.denselayer1.norm1.bias', 'features.denseblock3.denselayer1.norm1.running_mean', 'features.denseblock3.denselayer1.norm1.running_var', 'features.denseblock3.denselayer1.conv1.weight', 'features.denseblock3.denselayer1.norm2.weight', 'features.denseblock3.denselayer1.norm2.bias', 'features.denseblock3.denselayer1.norm2.running_mean', 'features.denseblock3.denselayer1.norm2.running_var', 'features.denseblock3.denselayer1.conv2.weight', 'features.denseblock3.denselayer2.norm1.weight', 'features.denseblock3.denselayer2.norm1.bias', 'features.denseblock3.denselayer2.norm1.running_mean', 'features.denseblock3.denselayer2.norm1.running_var', 'features.denseblock3.denselayer2.conv1.weight', 'features.denseblock3.denselayer2.norm2.weight', 'features.denseblock3.denselayer2.norm2.bias', 'features.denseblock3.denselayer2.norm2.running_mean', 'features.denseblock3.denselayer2.norm2.running_var', 'features.denseblock3.denselayer2.conv2.weight', 'features.denseblock3.denselayer3.norm1.weight', 'features.denseblock3.denselayer3.norm1.bias', 'features.denseblock3.denselayer3.norm1.running_mean', 'features.denseblock3.denselayer3.norm1.running_var', 'features.denseblock3.denselayer3.conv1.weight', 'features.denseblock3.denselayer3.norm2.weight', 'features.denseblock3.denselayer3.norm2.bias', 'features.denseblock3.denselayer3.norm2.running_mean', 'features.denseblock3.denselayer3.norm2.running_var', 'features.denseblock3.denselayer3.conv2.weight', 'features.denseblock3.denselayer4.norm1.weight', 'features.denseblock3.denselayer4.norm1.bias', 'features.denseblock3.denselayer4.norm1.running_mean', 'features.denseblock3.denselayer4.norm1.running_var', 'features.denseblock3.denselayer4.conv1.weight', 'features.denseblock3.denselayer4.norm2.weight', 'features.denseblock3.denselayer4.norm2.bias', 'features.denseblock3.denselayer4.norm2.running_mean', 'features.denseblock3.denselayer4.norm2.running_var', 'features.denseblock3.denselayer4.conv2.weight', 'features.denseblock3.denselayer5.norm1.weight', 'features.denseblock3.denselayer5.norm1.bias', 'features.denseblock3.denselayer5.norm1.running_mean', 'features.denseblock3.denselayer5.norm1.running_var', 'features.denseblock3.denselayer5.conv1.weight', 'features.denseblock3.denselayer5.norm2.weight', 'features.denseblock3.denselayer5.norm2.bias', 'features.denseblock3.denselayer5.norm2.running_mean', 'features.denseblock3.denselayer5.norm2.running_var', 'features.denseblock3.denselayer5.conv2.weight', 'features.denseblock3.denselayer6.norm1.weight', 'features.denseblock3.denselayer6.norm1.bias', 'features.denseblock3.denselayer6.norm1.running_mean', 'features.denseblock3.denselayer6.norm1.running_var', 'features.denseblock3.denselayer6.conv1.weight', 'features.denseblock3.denselayer6.norm2.weight', 'features.denseblock3.denselayer6.norm2.bias', 'features.denseblock3.denselayer6.norm2.running_mean', 'features.denseblock3.denselayer6.norm2.running_var', 'features.denseblock3.denselayer6.conv2.weight', 'features.denseblock3.denselayer7.norm1.weight', 'features.denseblock3.denselayer7.norm1.bias', 'features.denseblock3.denselayer7.norm1.running_mean', 'features.denseblock3.denselayer7.norm1.running_var', 'features.denseblock3.denselayer7.conv1.weight', 'features.denseblock3.denselayer7.norm2.weight', 'features.denseblock3.denselayer7.norm2.bias', 'features.denseblock3.denselayer7.norm2.running_mean', 'features.denseblock3.denselayer7.norm2.running_var', 'features.denseblock3.denselayer7.conv2.weight', 'features.denseblock3.denselayer8.norm1.weight', 'features.denseblock3.denselayer8.norm1.bias', 'features.denseblock3.denselayer8.norm1.running_mean', 'features.denseblock3.denselayer8.norm1.running_var', 'features.denseblock3.denselayer8.conv1.weight', 'features.denseblock3.denselayer8.norm2.weight', 'features.denseblock3.denselayer8.norm2.bias', 'features.denseblock3.denselayer8.norm2.running_mean', 'features.denseblock3.denselayer8.norm2.running_var', 'features.denseblock3.denselayer8.conv2.weight', 'features.denseblock3.denselayer9.norm1.weight', 'features.denseblock3.denselayer9.norm1.bias', 'features.denseblock3.denselayer9.norm1.running_mean', 'features.denseblock3.denselayer9.norm1.running_var', 'features.denseblock3.denselayer9.conv1.weight', 'features.denseblock3.denselayer9.norm2.weight', 'features.denseblock3.denselayer9.norm2.bias', 'features.denseblock3.denselayer9.norm2.running_mean', 'features.denseblock3.denselayer9.norm2.running_var', 'features.denseblock3.denselayer9.conv2.weight', 'features.denseblock3.denselayer10.norm1.weight', 'features.denseblock3.denselayer10.norm1.bias', 'features.denseblock3.denselayer10.norm1.running_mean', 'features.denseblock3.denselayer10.norm1.running_var', 'features.denseblock3.denselayer10.conv1.weight', 'features.denseblock3.denselayer10.norm2.weight', 'features.denseblock3.denselayer10.norm2.bias', 'features.denseblock3.denselayer10.norm2.running_mean', 'features.denseblock3.denselayer10.norm2.running_var', 'features.denseblock3.denselayer10.conv2.weight', 'features.denseblock3.denselayer11.norm1.weight', 'features.denseblock3.denselayer11.norm1.bias', 'features.denseblock3.denselayer11.norm1.running_mean', 'features.denseblock3.denselayer11.norm1.running_var', 'features.denseblock3.denselayer11.conv1.weight', 'features.denseblock3.denselayer11.norm2.weight', 'features.denseblock3.denselayer11.norm2.bias', 'features.denseblock3.denselayer11.norm2.running_mean', 'features.denseblock3.denselayer11.norm2.running_var', 'features.denseblock3.denselayer11.conv2.weight', 'features.denseblock3.denselayer12.norm1.weight', 'features.denseblock3.denselayer12.norm1.bias', 'features.denseblock3.denselayer12.norm1.running_mean', 'features.denseblock3.denselayer12.norm1.running_var', 'features.denseblock3.denselayer12.conv1.weight', 'features.denseblock3.denselayer12.norm2.weight', 'features.denseblock3.denselayer12.norm2.bias', 'features.denseblock3.denselayer12.norm2.running_mean', 'features.denseblock3.denselayer12.norm2.running_var', 'features.denseblock3.denselayer12.conv2.weight', 'features.denseblock3.denselayer13.norm1.weight', 'features.denseblock3.denselayer13.norm1.bias', 'features.denseblock3.denselayer13.norm1.running_mean', 'features.denseblock3.denselayer13.norm1.running_var', 'features.denseblock3.denselayer13.conv1.weight', 'features.denseblock3.denselayer13.norm2.weight', 'features.denseblock3.denselayer13.norm2.bias', 'features.denseblock3.denselayer13.norm2.running_mean', 'features.denseblock3.denselayer13.norm2.running_var', 'features.denseblock3.denselayer13.conv2.weight', 'features.denseblock3.denselayer14.norm1.weight', 'features.denseblock3.denselayer14.norm1.bias', 'features.denseblock3.denselayer14.norm1.running_mean', 'features.denseblock3.denselayer14.norm1.running_var', 'features.denseblock3.denselayer14.conv1.weight', 'features.denseblock3.denselayer14.norm2.weight', 'features.denseblock3.denselayer14.norm2.bias', 'features.denseblock3.denselayer14.norm2.running_mean', 'features.denseblock3.denselayer14.norm2.running_var', 'features.denseblock3.denselayer14.conv2.weight', 'features.denseblock3.denselayer15.norm1.weight', 'features.denseblock3.denselayer15.norm1.bias', 'features.denseblock3.denselayer15.norm1.running_mean', 'features.denseblock3.denselayer15.norm1.running_var', 'features.denseblock3.denselayer15.conv1.weight', 'features.denseblock3.denselayer15.norm2.weight', 'features.denseblock3.denselayer15.norm2.bias', 'features.denseblock3.denselayer15.norm2.running_mean', 'features.denseblock3.denselayer15.norm2.running_var', 'features.denseblock3.denselayer15.conv2.weight', 'features.denseblock3.denselayer16.norm1.weight', 'features.denseblock3.denselayer16.norm1.bias', 'features.denseblock3.denselayer16.norm1.running_mean', 'features.denseblock3.denselayer16.norm1.running_var', 'features.denseblock3.denselayer16.conv1.weight', 'features.denseblock3.denselayer16.norm2.weight', 'features.denseblock3.denselayer16.norm2.bias', 'features.denseblock3.denselayer16.norm2.running_mean', 'features.denseblock3.denselayer16.norm2.running_var', 'features.denseblock3.denselayer16.conv2.weight', 'features.denseblock3.denselayer17.norm1.weight', 'features.denseblock3.denselayer17.norm1.bias', 'features.denseblock3.denselayer17.norm1.running_mean', 'features.denseblock3.denselayer17.norm1.running_var', 'features.denseblock3.denselayer17.conv1.weight', 'features.denseblock3.denselayer17.norm2.weight', 'features.denseblock3.denselayer17.norm2.bias', 'features.denseblock3.denselayer17.norm2.running_mean', 'features.denseblock3.denselayer17.norm2.running_var', 'features.denseblock3.denselayer17.conv2.weight', 'features.denseblock3.denselayer18.norm1.weight', 'features.denseblock3.denselayer18.norm1.bias', 'features.denseblock3.denselayer18.norm1.running_mean', 'features.denseblock3.denselayer18.norm1.running_var', 'features.denseblock3.denselayer18.conv1.weight', 'features.denseblock3.denselayer18.norm2.weight', 'features.denseblock3.denselayer18.norm2.bias', 'features.denseblock3.denselayer18.norm2.running_mean', 'features.denseblock3.denselayer18.norm2.running_var', 'features.denseblock3.denselayer18.conv2.weight', 'features.denseblock3.denselayer19.norm1.weight', 'features.denseblock3.denselayer19.norm1.bias', 'features.denseblock3.denselayer19.norm1.running_mean', 'features.denseblock3.denselayer19.norm1.running_var', 'features.denseblock3.denselayer19.conv1.weight', 'features.denseblock3.denselayer19.norm2.weight', 'features.denseblock3.denselayer19.norm2.bias', 'features.denseblock3.denselayer19.norm2.running_mean', 'features.denseblock3.denselayer19.norm2.running_var', 'features.denseblock3.denselayer19.conv2.weight', 'features.denseblock3.denselayer20.norm1.weight', 'features.denseblock3.denselayer20.norm1.bias', 'features.denseblock3.denselayer20.norm1.running_mean', 'features.denseblock3.denselayer20.norm1.running_var', 'features.denseblock3.denselayer20.conv1.weight', 'features.denseblock3.denselayer20.norm2.weight', 'features.denseblock3.denselayer20.norm2.bias', 'features.denseblock3.denselayer20.norm2.running_mean', 'features.denseblock3.denselayer20.norm2.running_var', 'features.denseblock3.denselayer20.conv2.weight', 'features.denseblock3.denselayer21.norm1.weight', 'features.denseblock3.denselayer21.norm1.bias', 'features.denseblock3.denselayer21.norm1.running_mean', 'features.denseblock3.denselayer21.norm1.running_var', 'features.denseblock3.denselayer21.conv1.weight', 'features.denseblock3.denselayer21.norm2.weight', 'features.denseblock3.denselayer21.norm2.bias', 'features.denseblock3.denselayer21.norm2.running_mean', 'features.denseblock3.denselayer21.norm2.running_var', 'features.denseblock3.denselayer21.conv2.weight', 'features.denseblock3.denselayer22.norm1.weight', 'features.denseblock3.denselayer22.norm1.bias', 'features.denseblock3.denselayer22.norm1.running_mean', 'features.denseblock3.denselayer22.norm1.running_var', 'features.denseblock3.denselayer22.conv1.weight', 'features.denseblock3.denselayer22.norm2.weight', 'features.denseblock3.denselayer22.norm2.bias', 'features.denseblock3.denselayer22.norm2.running_mean', 'features.denseblock3.denselayer22.norm2.running_var', 'features.denseblock3.denselayer22.conv2.weight', 'features.denseblock3.denselayer23.norm1.weight', 'features.denseblock3.denselayer23.norm1.bias', 'features.denseblock3.denselayer23.norm1.running_mean', 'features.denseblock3.denselayer23.norm1.running_var', 'features.denseblock3.denselayer23.conv1.weight', 'features.denseblock3.denselayer23.norm2.weight', 'features.denseblock3.denselayer23.norm2.bias', 'features.denseblock3.denselayer23.norm2.running_mean', 'features.denseblock3.denselayer23.norm2.running_var', 'features.denseblock3.denselayer23.conv2.weight', 'features.denseblock3.denselayer24.norm1.weight', 'features.denseblock3.denselayer24.norm1.bias', 'features.denseblock3.denselayer24.norm1.running_mean', 'features.denseblock3.denselayer24.norm1.running_var', 'features.denseblock3.denselayer24.conv1.weight', 'features.denseblock3.denselayer24.norm2.weight', 'features.denseblock3.denselayer24.norm2.bias', 'features.denseblock3.denselayer24.norm2.running_mean', 'features.denseblock3.denselayer24.norm2.running_var', 'features.denseblock3.denselayer24.conv2.weight', 'features.transition3.norm.weight', 'features.transition3.norm.bias', 'features.transition3.norm.running_mean', 'features.transition3.norm.running_var', 'features.transition3.conv.weight', 'features.denseblock4.denselayer1.norm1.weight', 'features.denseblock4.denselayer1.norm1.bias', 'features.denseblock4.denselayer1.norm1.running_mean', 'features.denseblock4.denselayer1.norm1.running_var', 'features.denseblock4.denselayer1.conv1.weight', 'features.denseblock4.denselayer1.norm2.weight', 'features.denseblock4.denselayer1.norm2.bias', 'features.denseblock4.denselayer1.norm2.running_mean', 'features.denseblock4.denselayer1.norm2.running_var', 'features.denseblock4.denselayer1.conv2.weight', 'features.denseblock4.denselayer2.norm1.weight', 'features.denseblock4.denselayer2.norm1.bias', 'features.denseblock4.denselayer2.norm1.running_mean', 'features.denseblock4.denselayer2.norm1.running_var', 'features.denseblock4.denselayer2.conv1.weight', 'features.denseblock4.denselayer2.norm2.weight', 'features.denseblock4.denselayer2.norm2.bias', 'features.denseblock4.denselayer2.norm2.running_mean', 'features.denseblock4.denselayer2.norm2.running_var', 'features.denseblock4.denselayer2.conv2.weight', 'features.denseblock4.denselayer3.norm1.weight', 'features.denseblock4.denselayer3.norm1.bias', 'features.denseblock4.denselayer3.norm1.running_mean', 'features.denseblock4.denselayer3.norm1.running_var', 'features.denseblock4.denselayer3.conv1.weight', 'features.denseblock4.denselayer3.norm2.weight', 'features.denseblock4.denselayer3.norm2.bias', 'features.denseblock4.denselayer3.norm2.running_mean', 'features.denseblock4.denselayer3.norm2.running_var', 'features.denseblock4.denselayer3.conv2.weight', 'features.denseblock4.denselayer4.norm1.weight', 'features.denseblock4.denselayer4.norm1.bias', 'features.denseblock4.denselayer4.norm1.running_mean', 'features.denseblock4.denselayer4.norm1.running_var', 'features.denseblock4.denselayer4.conv1.weight', 'features.denseblock4.denselayer4.norm2.weight', 'features.denseblock4.denselayer4.norm2.bias', 'features.denseblock4.denselayer4.norm2.running_mean', 'features.denseblock4.denselayer4.norm2.running_var', 'features.denseblock4.denselayer4.conv2.weight', 'features.denseblock4.denselayer5.norm1.weight', 'features.denseblock4.denselayer5.norm1.bias', 'features.denseblock4.denselayer5.norm1.running_mean', 'features.denseblock4.denselayer5.norm1.running_var', 'features.denseblock4.denselayer5.conv1.weight', 'features.denseblock4.denselayer5.norm2.weight', 'features.denseblock4.denselayer5.norm2.bias', 'features.denseblock4.denselayer5.norm2.running_mean', 'features.denseblock4.denselayer5.norm2.running_var', 'features.denseblock4.denselayer5.conv2.weight', 'features.denseblock4.denselayer6.norm1.weight', 'features.denseblock4.denselayer6.norm1.bias', 'features.denseblock4.denselayer6.norm1.running_mean', 'features.denseblock4.denselayer6.norm1.running_var', 'features.denseblock4.denselayer6.conv1.weight', 'features.denseblock4.denselayer6.norm2.weight', 'features.denseblock4.denselayer6.norm2.bias', 'features.denseblock4.denselayer6.norm2.running_mean', 'features.denseblock4.denselayer6.norm2.running_var', 'features.denseblock4.denselayer6.conv2.weight', 'features.denseblock4.denselayer7.norm1.weight', 'features.denseblock4.denselayer7.norm1.bias', 'features.denseblock4.denselayer7.norm1.running_mean', 'features.denseblock4.denselayer7.norm1.running_var', 'features.denseblock4.denselayer7.conv1.weight', 'features.denseblock4.denselayer7.norm2.weight', 'features.denseblock4.denselayer7.norm2.bias', 'features.denseblock4.denselayer7.norm2.running_mean', 'features.denseblock4.denselayer7.norm2.running_var', 'features.denseblock4.denselayer7.conv2.weight', 'features.denseblock4.denselayer8.norm1.weight', 'features.denseblock4.denselayer8.norm1.bias', 'features.denseblock4.denselayer8.norm1.running_mean', 'features.denseblock4.denselayer8.norm1.running_var', 'features.denseblock4.denselayer8.conv1.weight', 'features.denseblock4.denselayer8.norm2.weight', 'features.denseblock4.denselayer8.norm2.bias', 'features.denseblock4.denselayer8.norm2.running_mean', 'features.denseblock4.denselayer8.norm2.running_var', 'features.denseblock4.denselayer8.conv2.weight', 'features.denseblock4.denselayer9.norm1.weight', 'features.denseblock4.denselayer9.norm1.bias', 'features.denseblock4.denselayer9.norm1.running_mean', 'features.denseblock4.denselayer9.norm1.running_var', 'features.denseblock4.denselayer9.conv1.weight', 'features.denseblock4.denselayer9.norm2.weight', 'features.denseblock4.denselayer9.norm2.bias', 'features.denseblock4.denselayer9.norm2.running_mean', 'features.denseblock4.denselayer9.norm2.running_var', 'features.denseblock4.denselayer9.conv2.weight', 'features.denseblock4.denselayer10.norm1.weight', 'features.denseblock4.denselayer10.norm1.bias', 'features.denseblock4.denselayer10.norm1.running_mean', 'features.denseblock4.denselayer10.norm1.running_var', 'features.denseblock4.denselayer10.conv1.weight', 'features.denseblock4.denselayer10.norm2.weight', 'features.denseblock4.denselayer10.norm2.bias', 'features.denseblock4.denselayer10.norm2.running_mean', 'features.denseblock4.denselayer10.norm2.running_var', 'features.denseblock4.denselayer10.conv2.weight', 'features.denseblock4.denselayer11.norm1.weight', 'features.denseblock4.denselayer11.norm1.bias', 'features.denseblock4.denselayer11.norm1.running_mean', 'features.denseblock4.denselayer11.norm1.running_var', 'features.denseblock4.denselayer11.conv1.weight', 'features.denseblock4.denselayer11.norm2.weight', 'features.denseblock4.denselayer11.norm2.bias', 'features.denseblock4.denselayer11.norm2.running_mean', 'features.denseblock4.denselayer11.norm2.running_var', 'features.denseblock4.denselayer11.conv2.weight', 'features.denseblock4.denselayer12.norm1.weight', 'features.denseblock4.denselayer12.norm1.bias', 'features.denseblock4.denselayer12.norm1.running_mean', 'features.denseblock4.denselayer12.norm1.running_var', 'features.denseblock4.denselayer12.conv1.weight', 'features.denseblock4.denselayer12.norm2.weight', 'features.denseblock4.denselayer12.norm2.bias', 'features.denseblock4.denselayer12.norm2.running_mean', 'features.denseblock4.denselayer12.norm2.running_var', 'features.denseblock4.denselayer12.conv2.weight', 'features.denseblock4.denselayer13.norm1.weight', 'features.denseblock4.denselayer13.norm1.bias', 'features.denseblock4.denselayer13.norm1.running_mean', 'features.denseblock4.denselayer13.norm1.running_var', 'features.denseblock4.denselayer13.conv1.weight', 'features.denseblock4.denselayer13.norm2.weight', 'features.denseblock4.denselayer13.norm2.bias', 'features.denseblock4.denselayer13.norm2.running_mean', 'features.denseblock4.denselayer13.norm2.running_var', 'features.denseblock4.denselayer13.conv2.weight', 'features.denseblock4.denselayer14.norm1.weight', 'features.denseblock4.denselayer14.norm1.bias', 'features.denseblock4.denselayer14.norm1.running_mean', 'features.denseblock4.denselayer14.norm1.running_var', 'features.denseblock4.denselayer14.conv1.weight', 'features.denseblock4.denselayer14.norm2.weight', 'features.denseblock4.denselayer14.norm2.bias', 'features.denseblock4.denselayer14.norm2.running_mean', 'features.denseblock4.denselayer14.norm2.running_var', 'features.denseblock4.denselayer14.conv2.weight', 'features.denseblock4.denselayer15.norm1.weight', 'features.denseblock4.denselayer15.norm1.bias', 'features.denseblock4.denselayer15.norm1.running_mean', 'features.denseblock4.denselayer15.norm1.running_var', 'features.denseblock4.denselayer15.conv1.weight', 'features.denseblock4.denselayer15.norm2.weight', 'features.denseblock4.denselayer15.norm2.bias', 'features.denseblock4.denselayer15.norm2.running_mean', 'features.denseblock4.denselayer15.norm2.running_var', 'features.denseblock4.denselayer15.conv2.weight', 'features.denseblock4.denselayer16.norm1.weight', 'features.denseblock4.denselayer16.norm1.bias', 'features.denseblock4.denselayer16.norm1.running_mean', 'features.denseblock4.denselayer16.norm1.running_var', 'features.denseblock4.denselayer16.conv1.weight', 'features.denseblock4.denselayer16.norm2.weight', 'features.denseblock4.denselayer16.norm2.bias', 'features.denseblock4.denselayer16.norm2.running_mean', 'features.denseblock4.denselayer16.norm2.running_var', 'features.denseblock4.denselayer16.conv2.weight', 'features.norm5.weight', 'features.norm5.bias', 'features.norm5.running_mean', 'features.norm5.running_var', 'classifier.fc1.weight', 'classifier.fc1.bias', 'classifier.fc2.weight', 'classifier.fc2.bias'])



```python
model.load_state_dict(state_dict)
```


```python
index = 3

model.eval()

images, label = next(iter(trainloader))
helper.imshow(images[index,:])

image = images.to(device)

# Calculate the class probabilities (softmax) for img
with torch.no_grad():
    output = model.forward(image)

ps = torch.exp(output)
ps[index]
# Plot the image and probabilities
# helper.view_classify(img.view(1, 224, 224), ps)
```




    tensor([ 0.0136,  0.9864], device='cuda:0')




![png](output_18_1.png)



```python
images, label = next(iter(trainloader))

helper.imshow(images[5,:])

image = images.to(device)

# Calculate the class probabilities (softmax) for img
with torch.no_grad():
    output = model.forward(image)

ps = torch.exp(output)
ps[5]
```




    tensor([ 9.9999e-01,  6.8896e-06], device='cuda:0')




![png](output_19_1.png)

