import time
from dataclasses import dataclass
from typing import Literal, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
from tqdm.notebook import tqdm


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()
    return correct / total


def backward_pass(classifier, inputs, labels, criterion):
    outputs = classifier.model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    classifier.optimizer.step()
    return outputs, loss


@dataclass
class TrainParams:
    seed: int
    architecture: Literal["resnet18", "resnet34", "resnet50"]
    optimizer: Literal["adam", "nag"]
    freeze_layers: bool
    # Epochs (unordered) at which we unfreeze the second-to-last layer, then third-to-last layer
    unfreezing_epochs: Tuple[int, ...]
    # Number of times per epoch validation accuracy is recorded
    validation_freq: Optional[int]
    # Computational budget in seconds -- aborts training when the time limit is exceeded
    time_limit_seconds: Optional[int]
    # Stop training if this validation accuracy is exceeded during training
    val_acc_target: Optional[float]


class Classifier:
    num_classes = 37
    arch_dict = dict(
        resnet18=(ResNet18_Weights.DEFAULT, models.resnet18),
        resnet34=(ResNet34_Weights.DEFAULT, models.resnet34),
        resnet50=(ResNet50_Weights.DEFAULT, models.resnet50),
    )

    def __init__(self, params: TrainParams):
        self.device = self._make_device()
        self.transform = self._make_transform(params)
        self.model = self._make_model(params, self.device)
        self.optimizer = self._make_optimizer(params, self.model)
        self.epoch_to_unfreezing = self._make_unfreezings(params, self.model)
        self.params = params
        self.training_start = None
        self.validation_accuracies = []
        self.training_accuracies = []
        self.epoch_losses = []

    @classmethod
    def _make_model(cls, params: TrainParams, device):
        weights, model_fn = Classifier.arch_dict[params.architecture]
        model = model_fn(weights=weights)

        if params.freeze_layers:
            for param in model.parameters():
                param.requires_grad = False  # Freeze all layers
        for param in model.fc.parameters():
            param.requires_grad = True  # Unfreeze final layer

        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, Classifier.num_classes)
        return model.to(device)

    @classmethod
    def _make_optimizer(cls, params, model):
        if params.optimizer == "adam":
            return optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=0.001
            )
        elif params.optimizer == "nag":
            return optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=0.01,
                momentum=0.9,
                nesterov=True,
                weight_decay=1e-4,
            )
        else:
            raise NotImplementedError

    def remake_optimizer(self, model):
        return self._make_optimizer(self.params, model)

    @classmethod
    def _make_device(cls):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def _make_transform(cls, params: TrainParams):
        weights, _ = Classifier.arch_dict[params.architecture]
        return weights.transforms()

    def gpu_acceleration_enabled(self):
        return self.device.type == 'cuda'

    def start_training(self):
        self.training_start = time.perf_counter()

    @classmethod
    def _make_unfreezings(cls, params: TrainParams, model):
        # Define layers to gradually unfreeze
        layer_names = [model.layer4, model.layer3]  # layer4 = last block
        if params.unfreezing_epochs in {None, ()}:
            return dict()
        unfreezing_epochs = sorted(params.unfreezing_epochs)
        if len(unfreezing_epochs) == 1:
            return {
                unfreezing_epochs[0]: {layer_names[0]}
            }
        if len(unfreezing_epochs) == 2:
            return {
                unfreezing_epochs[0]: ("layer4", layer_names[0]),
                unfreezing_epochs[1]: ("layer3", layer_names[1]),
            }
        else:
            raise NotImplementedError()

    def maybe_unfreeze(self, epoch: int):
        if epoch not in self.epoch_to_unfreezing.keys():
            return
        layer_label, layer = self.epoch_to_unfreezing[epoch]
        for param in layer.parameters():
            param.requires_grad = True
        print(f"Unfroze {layer_label} at epoch {epoch}")
        # Recreate optimizer after unfreezing more layers
        self.optimizer = self._make_optimizer(self.params, self.model)

    def should_record_metrics(self) -> bool:
        if self.params.validation_freq == 1:
            return True
        elif self.params.validation_freq == 0:
            return False
        else:
            raise NotImplementedError()

    def should_stop_training_early(self) -> bool:
        current_val_acc = self.validation_accuracies[-1]
        val_acc_target = self.params.val_acc_target
        if val_acc_target is not None and current_val_acc >= val_acc_target:
            print(f"Exceeded target validation accuracy -- stopping training.")
            return True
        running_time_seconds = time.perf_counter() - self.training_start
        time_limit_seconds = self.params.time_limit_seconds
        if time_limit_seconds is not None and running_time_seconds >= time_limit_seconds:
            time_limit_seconds = self.params.time_limit_seconds
            print(
                f"Exhausted {running_time_seconds:.0f}/{time_limit_seconds} seconds of the computational budget -> stopping training.")
            return True
        return False
