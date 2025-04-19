#!/usr/bin/env python3

import torchvision.models as models
from torchvision.models import ResNet18_Weights


def main():
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)


if __name__ == '__main__':
    main()
