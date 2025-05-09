import copy
import time
import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Literal, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
from torchvision.transforms import transforms
from tqdm.auto import tqdm


from util import dumps_inline_lists


def evaluate(model, device, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
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
    weight_decay: float
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
    # Finetune l layers simultaneously
    unfreeze_last_l_blocks: Optional[int] = None
    # data augmentation
    data_augmentation: Optional[Literal["true", "false"]] = None
    # Unsupervised learning params
    unsup_weight: Optional[float] = 0.5
    psuedo_threshold: Optional[float] = 0.95
    

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

    def pprint(self):
        return dumps_inline_lists(self.minimal_dict())

    def copy(self) -> 'TrainParams':
        return copy.deepcopy(self)


@dataclass
class TrainingResult:
    training_losses: Tuple[float, ...]
    training_accuracies: Tuple[float, ...]
    validation_accuracies: Tuple[float, ...]
    update_steps: Tuple[int, ...]
    epochs: Tuple[int, ...]
    training_elapsed: float

    def pprint(self):
        return dumps_inline_lists(asdict(self))


class StopCondition(ABC):
    @abstractmethod
    def remaining_steps(self, trainer: 'Trainer') -> int:
        """ How many update steps do we expect to run for? Assumed to be called before the loop. """
        raise NotImplementedError()

    @abstractmethod
    def should_stop(self, trainer: 'Trainer') -> bool:
        """ Iff true, the training session ends. """
        raise NotImplementedError()


class FinishedAllEpochs(StopCondition):
    def remaining_steps(self, trainer: 'Trainer') -> int:
        max_num_epochs = trainer.params.n_epochs
        max_total_update_steps = max_num_epochs * len(trainer.labelled_train_loader)
        return max_total_update_steps

    def should_stop(self, trainer: 'Trainer') -> bool:
        return trainer.epoch >= trainer.params.n_epochs


class FinishedEpochs(StopCondition):
    def __init__(self, epoch_count):
        self.epoch_count = epoch_count  # Number of epochs to finish before stopping
        self.epoch_at_start_of_session = None

    def remaining_steps(self, trainer: 'Trainer') -> int:
        self.epoch_at_start_of_session = trainer.epoch
        remaining_epochs = self.epoch_count - (trainer.epoch - self.epoch_at_start_of_session)
        return remaining_epochs * len(trainer.train_loader)

    def should_stop(self, trainer: 'Trainer') -> bool:
        return trainer.epoch >= self.epoch_at_start_of_session + self.epoch_count


class Trainer:
    num_classes = 37
    arch_dict = dict(
        resnet18=(ResNet18_Weights.DEFAULT, models.resnet18),
        resnet34=(ResNet34_Weights.DEFAULT, models.resnet34),
        resnet50=(ResNet50_Weights.DEFAULT, models.resnet50),
    )

    def __init__(self, params: TrainParams, verbose=True):
        self.verbose = verbose
        self.device = self._make_device()
        self.transform = self.make_transform(params)
        self.model = self._make_model(params, self.device)
        self.optimizer = self._make_optimizer(params, self.model)
        self.epoch_to_unfreezing = self._make_unfreezings(params, self.model)
        self.params = params
        self.epoch = 0  # Current epoch. 0 means training has not yet begun. Starts at 1.
        self.update_step = 0  # Current update step. 0 means training has not yet begun. Starts at 1.

        self.validation_accuracies = []
        self.training_accuracies = []
        self.training_losses = []
        self.recorded_update_steps = []
        self.labelled_train_loader = None
        self.unlabelled_train_loader = None
        self.val_loader = None

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
                weight_decay=params.optimizer.weight_decay,
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
        if params.data_augmentation is not None and params.data_augmentation == "true":
            print("performing data augmentation")
            train_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),  # Random flip with probability 0.5
                transforms.RandomRotation(15),  # Random rotation within +/-15 degrees
                transforms.ToTensor()  # Convert to tensor
                #     transforms.Normalize(
                #         mean=[0.485, 0.456, 0.406],             # Imagenet mean
                #         std=[0.229, 0.224, 0.225]               # Imagenet std
                #     )
            ])
            return train_transforms
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
        # print(f"Unfroze {layer_label} at epoch {epoch}")
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

    def maybe_unfreeze_last_layers(self, l, model: nn.Module):
        if l is None:
            return
        # Define the model blocks (last to first)
        layer_blocks = [
            model.layer4,
            model.layer3,
            model.layer2,
            model.layer1,
            model.conv1,
            model.bn1,
        ]

        # Unfreeze the last l blocks
        for i in range(min(l, len(layer_blocks))):
            for param in layer_blocks[i].parameters():
                param.requires_grad = True

        print(f"[Trainer] Unfroze last {l} blocks")

    def load(self, labelled_train_loader, unlabelled_train_loader, val_loader):
        self.labelled_train_loader = labelled_train_loader
        self.unlabelled_train_loader = unlabelled_train_loader
        self.val_loader = val_loader

    def train(self, stop_condition: StopCondition) -> TrainingResult:
        training_start = time.perf_counter()
        if self.labelled_train_loader is None or self.val_loader is None:
            raise ValueError("Must call Trainer.load(...) before training")
        criterion = nn.CrossEntropyLoss()
        model = self.model

        max_num_epochs = self.params.n_epochs
        num_epochs = max_num_epochs
        model.train()

        self.maybe_unfreeze_last_layers(self.params.unfreeze_last_l_blocks, model)

        # pseudo-labelling 
        if self.unlabelled_train_loader is not None: 
            unlabelled_iter = itertools.cycle(self.unlabelled_train_loader)
            unsup_weight = self.params.unsup_weight
            psuedo_threshold = self.params.psuedo_threshold

        remaining_steps = stop_condition.remaining_steps(self)
        if self.verbose:
            pb_update_steps = tqdm(range(1, remaining_steps + 1), desc="Update step", leave=True)  # Progress bar

        while not stop_condition.should_stop(self):
            self.epoch += 1
            running_loss = 0.0
            correct = 0
            total = 0

            self.maybe_unfreeze(self.epoch)

            for inputs, labels in self.labelled_train_loader:
                if self.verbose:
                    pb_update_steps.update(1)  # Move progress bar by 1
                self.update_step += 1
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs, loss = backward_pass(self, inputs, labels, criterion)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)

                #psuedo-labelling
                if self.unlabelled_train_loader is not None:
                    batch_u = next(unlabelled_iter)
                    if isinstance(batch_u, (list, tuple)):
                        x_u = batch_u[0]
                    else:
                        x_u = batch_u
                    x_u = x_u.to(self.device)
                    with torch.no_grad():
                        logits_u = model(x_u)
                        probs_u, pseudo = F.softmax(logits_u, dim=1).max(1)
                        mask = probs_u.ge(psuedo_threshold)

                    if mask.any():
                        inputs_u  = x_u[mask]
                        labels_u  = pseudo[mask]
                        _, unsup_loss = backward_pass(self, inputs_u, labels_u, criterion)
                        running_loss += unsup_weight * unsup_loss.item()
                    
                
                correct += (predicted == labels).sum().item()

            train_acc = correct / total

            should_record_metrics = self.should_record_metrics()
            if should_record_metrics:
                val_acc = evaluate(model, self.device, self.val_loader)
                val_acc_str = f", Val Acc: {100 * val_acc:.2f}%"
            else:
                val_acc = None
                val_acc_str = ""
            self.training_losses.append(running_loss / len(self.labelled_train_loader))
            self.training_accuracies.append(train_acc)
            if should_record_metrics:
                self.recorded_update_steps.append(self.update_step)
                self.validation_accuracies.append(val_acc)

            if self.verbose:
                pb_update_steps.refresh()
                tqdm.write(
                    f"Epoch [{self.epoch}/{max_num_epochs}], Loss: {running_loss / len(self.labelled_train_loader):.4f}, Train Acc: {100 * train_acc:.2f}%{val_acc_str}")

            if self.should_stop_training_early(self.validation_accuracies, training_start):
                num_epochs = self.epoch
                break

        if self.verbose:
            tqdm.write(
                f"Total elapsed: {pb_update_steps.format_dict['elapsed']:.2f}s, average per update step: {1 / pb_update_steps.format_dict['rate']:.2f}s")
            pb_update_steps.close()

        # Shouldn't be necessary, should prevent tqdm-related hanging if it appears
        terminate_workers(self.labelled_train_loader, self.unlabelled_train_loader, self.val_loader)

        epochs = range(1, num_epochs + 1)
        training_elapsed = time.perf_counter() - training_start
        return TrainingResult(
            training_losses=tuple(self.training_losses),
            training_accuracies=tuple(self.training_accuracies),
            validation_accuracies=tuple(self.validation_accuracies),
            epochs=tuple(epochs),
            update_steps=tuple(self.recorded_update_steps),
            training_elapsed=training_elapsed,
        )


def terminate_workers(labelled_train_loader, unlabelled_train_loader, val_loader):
    for loader in (labelled_train_loader, unlabelled_train_loader, val_loader):
        it = getattr(loader, "_iterator", None)
        if it is not None:
            it._shutdown_workers()
            del loader._iterator
