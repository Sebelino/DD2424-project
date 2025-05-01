import time
from dataclasses import dataclass, asdict
from typing import Literal, Tuple, Optional, Any

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


def backward_pass(trainer, inputs, labels, criterion):
    outputs = trainer.model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    trainer.optimizer.step()
    return outputs, loss


@dataclass
class AdamParams:
    learning_rate: float
    name: str = "adam"


@dataclass
class NagParams:
    learning_rate: float
    weight_decay: float
    momentum: float
    name: str = "nag"


@dataclass
class TrainParams:
    seed: int
    batch_size: int
    architecture: Literal["resnet18", "resnet34", "resnet50"]
    optimizer: NagParams | AdamParams
    n_epochs: int
    freeze_layers: bool
    # Epochs (unordered) at which we unfreeze the second-to-last layer, then third-to-last layer
    unfreezing_epochs: Tuple[int, ...]
    # Number of times per epoch validation accuracy is recorded
    validation_freq: Optional[int]
    # Computational budget in seconds -- aborts training when the time limit is exceeded
    time_limit_seconds: Optional[int]
    # Stop training if this validation accuracy is exceeded during training
    val_acc_target: Optional[float]

    def minimal_dict(self) -> dict[str, Any]:
        def prune(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {
                    key: prune(val)
                    for key, val in obj.items()
                    if val is not None
                }
            if isinstance(obj, (list, tuple)):
                pruned = [prune(val) for val in obj if val is not None]
                return type(obj)(pruned)
            return obj

        return prune(asdict(self))


@dataclass
class TrainingResult:
    training_losses: Tuple[float, ...]
    training_accuracies: Tuple[float, ...]
    validation_accuracies: Tuple[float, ...]
    update_steps: Tuple[int, ...]
    epochs: Tuple[int, ...]
    training_elapsed: float


class Trainer:
    num_classes = 37
    arch_dict = dict(
        resnet18=(ResNet18_Weights.DEFAULT, models.resnet18),
        resnet34=(ResNet34_Weights.DEFAULT, models.resnet34),
        resnet50=(ResNet50_Weights.DEFAULT, models.resnet50),
    )

    def __init__(self, params: TrainParams):
        self.device = self._make_device()
        self.transform = self.make_transform(params)
        self.model = self._make_model(params, self.device)
        self.optimizer = self._make_optimizer(params, self.model)
        self.epoch_to_unfreezing = self._make_unfreezings(params, self.model)
        self.params = params

    @classmethod
    def _make_model(cls, params: TrainParams, device):
        weights, model_fn = Trainer.arch_dict[params.architecture]
        model = model_fn(weights=weights)

        if params.freeze_layers:
            for param in model.parameters():
                param.requires_grad = False  # Freeze all layers
        for param in model.fc.parameters():
            param.requires_grad = True  # Unfreeze final layer

        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, Trainer.num_classes)
        return model.to(device)

    @classmethod
    def _make_optimizer(cls, params, model):
        if params.optimizer.name == "adam":
            return optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=params.optimizer.learning_rate,
            )
        elif params.optimizer.name == "nag":
            return optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=params.optimizer.learning_rate,
                momentum=params.optimizer.momentum,
                nesterov=True,
                weight_decay=params.optimizer.weight_decay,
            )
        else:
            raise NotImplementedError

    def remake_optimizer(self, model):
        return self._make_optimizer(self.params, model)

    @classmethod
    def _make_device(cls):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def make_transform(cls, params: TrainParams):
        weights, _ = cls.arch_dict[params.architecture]
        return weights.transforms()

    def gpu_acceleration_enabled(self):
        return self.device.type == 'cuda'

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

    def should_stop_training_early(self, validation_accuracies, training_start) -> bool:
        current_val_acc = validation_accuracies[-1]
        val_acc_target = self.params.val_acc_target
        if val_acc_target is not None and current_val_acc >= val_acc_target:
            print(f"Exceeded target validation accuracy -- stopping training.")
            return True
        running_time_seconds = time.perf_counter() - training_start
        time_limit_seconds = self.params.time_limit_seconds
        if time_limit_seconds is not None and running_time_seconds >= time_limit_seconds:
            time_limit_seconds = self.params.time_limit_seconds
            print(
                f"Exhausted {running_time_seconds:.0f}/{time_limit_seconds} seconds of the computational budget -> stopping training.")
            return True
        return False

    def train(self, train_loader, val_loader) -> TrainingResult:
        training_start = time.perf_counter()
        criterion = nn.CrossEntropyLoss()
        model = self.model

        max_num_epochs = self.params.n_epochs
        num_epochs = max_num_epochs
        model.train()

        validation_accuracies = []
        training_accuracies = []
        training_losses = []
        update_steps = []

        update_step = 1  # Epoch and update step start from 1
        progress_bar = tqdm(range(1, max_num_epochs + 1), desc="Epoch")
        for epoch in progress_bar:
            running_loss = 0.0
            correct = 0
            total = 0

            self.maybe_unfreeze(epoch)

            for inputs, labels in tqdm(train_loader, desc="Batch"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs, loss = backward_pass(self, inputs, labels, criterion)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                update_step += 1
            train_acc = correct / total

            should_record_metrics = self.should_record_metrics()
            if should_record_metrics:
                val_acc = evaluate(model, val_loader, self.device)
                val_acc_str = f", Val Acc: {100 * val_acc:.2f}%"
            else:
                val_acc = None
                val_acc_str = ""
            training_losses.append(running_loss / len(train_loader))
            training_accuracies.append(train_acc)
            if should_record_metrics:
                update_steps.append(update_step)
                validation_accuracies.append(val_acc)

            tqdm.write(
                f"Epoch [{epoch}/{max_num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Train Acc: {100 * train_acc:.2f}%{val_acc_str}")

            if self.should_stop_training_early(validation_accuracies, training_start):
                num_epochs = epoch
                break
        epochs = range(1, num_epochs + 1)
        training_elapsed = time.perf_counter() - training_start
        return TrainingResult(
            training_losses=tuple(training_losses),
            training_accuracies=tuple(training_accuracies),
            validation_accuracies=tuple(validation_accuracies),
            epochs=tuple(epochs),
            update_steps=tuple(update_steps),
            training_elapsed=training_elapsed,
        )
