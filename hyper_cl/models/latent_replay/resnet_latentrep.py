"""This is the slimmed ResNet as used by Lopez et al. in the GEM paper."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d
from typing import List
from itertools import chain

from avalanche.models import MultiHeadClassifier, MultiTaskModule


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> F.conv2d:
    """
    Instantiates a 3x3 convolutional layer with no bias.
    :param in_planes: number of input channels
    :param out_planes: number of output channels
    :param stride: stride of the convolution
    :return: convolutional layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """
    The basic block of ResNet.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        """
        Instantiates the basic block of the network.
        :param in_planes: the number of input channels
        :param planes: the number of channels (to be possibly expanded)
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (10)
        """
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class ResNetLatentReplay(nn.Module):
    """
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int, latent_depth=1) -> None:
        """
        Instantiates the layers of the network.
        :param block: the basic ResNet block
        :param num_blocks: the number of blocks per layer
        :param num_classes: the number of output classes
        :param nf: the number of filters
        """
        super(ResNetLatentReplay, self).__init__()
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.classifier = nn.Linear(nf * 8 * block.expansion, num_classes)

        self.latent_depth = latent_depth
        self.latent_params = [
            chain(self.conv1.parameters(), self.bn1.parameters()),
            self.layer1.parameters(),
            self.layer2.parameters(),
            self.layer3.parameters(),
            self.layer4.parameters(),
        ]

        self.latent_layers = [
            lambda x: relu(self.bn1(self.conv1(x))),
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4
        ]

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        """
        Instantiates a ResNet layer.
        :param block: ResNet basic block
        :param planes: channels across the network
        :param num_blocks: number of blocks
        :param stride: stride
        :return: ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def get_feat_params(self):
        return chain(*[self.latent_params[i]
                       for i in range(0, self.latent_depth)])

    def extract_feat(self, x):
        feat = x
        for i in range(self.latent_depth):
            feat = self.latent_layers[i](feat)

        return feat

    def forward_feat(self, x):
        feat = x
        for i in range(self.latent_depth, len(self.latent_layers)):
            feat = self.latent_layers[i](feat)

        out = avg_pool2d(feat, feat.shape[2])
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (output_classes)
        """
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, out.shape[2])
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class ResNet18LatentReplay(ResNetLatentReplay):
    """MultiTask Slimmed ResNet18."""

    def __init__(self, num_classes=10, nf=20, latent_depth=1):
        super().__init__(block=BasicBlock, num_blocks=[2, 2, 2, 2],
                         num_classes=num_classes, nf=nf,
                         latent_depth=latent_depth)


class MTResNet18LatentReplay(ResNet18LatentReplay, MultiTaskModule):
    """MultiTask Slimmed ResNet18."""

    def __init__(self, nf=20, latent_depth=1):
        super().__init__(nf=nf, latent_depth=latent_depth)
        self.classifier = MultiHeadClassifier(nf * 8 * BasicBlock.expansion, 2)

    def forward_feat(self, x, task_labels):
        feat = x
        for i in range(self.latent_depth, len(self.latent_layers)):
            feat = self.latent_layers[i](feat)

        out = avg_pool2d(feat, feat.shape[2])
        out = out.view(out.size(0), -1)
        out = self.classifier(out, task_labels)

        return out

    def forward(self, x, task_labels):
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :param task_labels: input tensor (batch_size)
        :return: output tensor (output_classes)
        """
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, out.shape[2])
        out = out.view(out.size(0), -1)
        out = self.classifier(out, task_labels)

        return out
