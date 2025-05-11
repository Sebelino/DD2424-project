from typing import Literal

from torchvision import transforms, models
from torchvision.models import ResNet50_Weights, ResNet34_Weights, ResNet18_Weights
from torchvision.transforms import InterpolationMode

arch_dict = dict(
    resnet18=(ResNet18_Weights.DEFAULT, models.resnet18),
    resnet34=(ResNet34_Weights.DEFAULT, models.resnet34),
    resnet50=(ResNet50_Weights.DEFAULT, models.resnet50),
)


def make_base_transform(architecture: Literal["resnet18", "resnet34", "resnet50"]):
    weights, _ = arch_dict[architecture]
    return weights.transforms()


def make_augmented_transform(base_tf):
    print("Performing data augmentation")
    return transforms.Compose([
        # 1) RandomResizedCrop *replaces* Resize→CenterCrop
        transforms.RandomResizedCrop(
            size=base_tf.crop_size,
            scale=(0.8, 1.0),
            ratio=(3 / 4, 4 / 3),
            #interpolation=base_tf.interpolation,
            interpolation=InterpolationMode.BILINEAR,
        ),
        # 2) your extra augs
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(
            degrees=15,
            expand=False,
            fill=tuple(int(255 * m) for m in base_tf.mean)  # fill with ImageNet mean
        ),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomGrayscale(p=0.1),
        # 3) final tensor + normalize
        transforms.ToTensor(),
        transforms.Normalize(base_tf.mean, base_tf.std),
    ])


def make_visualizable(tf, base_tf):
    mean, std = base_tf.mean, base_tf.std
    inv_norm = transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1 / s for s in std]
    )

    # 3) stitch them together
    return transforms.Compose([
        tf,
        inv_norm,
        transforms.ToPILImage(),
    ])
