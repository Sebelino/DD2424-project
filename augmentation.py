from dataclasses import dataclass
from typing import Literal

from torchvision import transforms, models
from torchvision.models import ResNet50_Weights, ResNet34_Weights, ResNet18_Weights
from torchvision.transforms import InterpolationMode

arch_dict = dict(
    resnet18=(ResNet18_Weights.DEFAULT, models.resnet18),
    resnet34=(ResNet34_Weights.DEFAULT, models.resnet34),
    resnet50=(ResNet50_Weights.DEFAULT, models.resnet50),
)


@dataclass
class AugmentationParams:
    enabled: bool
    transform: transforms.Compose

    def __reduce__(self):  # For pickle
        return (AugmentationParams, (self.enabled, self.transform))


def make_base_transform(architecture: Literal["resnet18", "resnet34", "resnet50"]):
    weights, _ = arch_dict[architecture]
    return weights.transforms()


def to_transform(architecture: Literal["resnet18", "resnet34", "resnet50"], aug_list: list[str]):
    base_tf = make_base_transform(architecture)
    augmentations = dict(
        resize=transforms.RandomResizedCrop(
            size=base_tf.crop_size,
            scale=(0.8, 1.0),
            ratio=(3 / 4, 4 / 3),
            interpolation=InterpolationMode.BILINEAR,
        ),
        flip=transforms.RandomHorizontalFlip(p=0.5),
        rotate=transforms.RandomRotation(
            degrees=15,
            expand=False,
            fill=tuple(int(255 * m) for m in base_tf.mean)  # fill with ImageNet mean
        ),
        colorjitter=transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        grayscale=transforms.RandomGrayscale(p=0.1),
    )
    compose_list = []
    for aug_str in aug_list:
        operation = augmentations[aug_str]
        compose_list.append(operation)
    compose_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(base_tf.mean, base_tf.std),
    ])
    return transforms.Compose(compose_list)


def auto_transform(architecture: Literal["resnet18", "resnet34", "resnet50"]):
    base_tf = make_base_transform(architecture)
    return transforms.Compose([
        transforms.RandomResizedCrop(
            size=base_tf.crop_size,
            scale=(0.8, 1.0),
            ratio=(3 / 4, 4 / 3),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(mean=base_tf.mean, std=base_tf.std),
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


def is_transform(obj):
    return isinstance(obj, transforms.Compose)


def serialize(obj: transforms.Compose) -> str:
    s = repr(obj)
    # s = s.replace("\n", " ")  # Improves printout. Generally error-prone but should be fine in our case.
    return s


def deserialize(s: str) -> transforms.Compose:
    safe_ns = {
        'Compose': transforms.Compose,
        'Resize': transforms.Resize,
        'CenterCrop': transforms.CenterCrop,
        'ToTensor': transforms.ToTensor,
        'BILINEAR': transforms.InterpolationMode.BILINEAR,
    }
    obj = eval(s, safe_ns)  # Unsafe but pragmatic
    return obj

def create_fixmatch_transforms():
    architecture = "resnet50"
    base_tf = make_base_transform(architecture)
    
    # Weak augmentation - just simple flip and small crop
    weak_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            size=base_tf.crop_size,
            scale=(0.8, 1.0),
            ratio=(3/4, 4/3),
            interpolation=InterpolationMode.BILINEAR,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(base_tf.mean, base_tf.std),
    ])
    
    # Strong augmentation - RandAugment + CTAugment as per FixMatch paper
    # Since RandAugment may not be directly available, we'll use a combination of transforms
    strong_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            size=base_tf.crop_size,
            scale=(0.8, 1.0),
            ratio=(3/4, 4/3),
            interpolation=InterpolationMode.BILINEAR,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),  # Stronger color jittering
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomRotation(
            degrees=30,  # Stronger rotation
            expand=False,
            fill=tuple(int(255 * m) for m in base_tf.mean)
        ),
        transforms.ToTensor(),
        transforms.Normalize(base_tf.mean, base_tf.std),
    ])
    
    return weak_transform, strong_transform