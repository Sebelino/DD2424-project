import copy
import hashlib
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from typing import Literal, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm

import augmentation
from datasets import DatasetParams
from determinism import Determinism
from freezing import compute_gradient_masks, apply_masks_and_freeze, MaskedFineTuningParams, make_unfreezings, \
    maybe_unfreeze_last_layers
from util import dumps_inline_lists, suppress_weights_only_warning


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
    # Data augmentation
    augmentation: augmentation.AugmentationParams
    # Number of times per epoch validation accuracy is recorded
    validation_freq: Optional[int] = 1
    # Computational budget in seconds -- aborts training when the time limit is exceeded
    time_limit_seconds: Optional[int] = None
    # Stop training if this validation accuracy is exceeded during training
    val_acc_target: Optional[float] = None
    # Masked fine-tuning
    #mft: MaskedFineTuningParams = MaskedFineTuningParams(enabled=False, k=0)
    mft: MaskedFineTuningParams = field(default_factory=lambda: MaskedFineTuningParams(enabled=False, k=0))
    # Finetune l layers simultaneously
    unfreeze_last_l_blocks: Optional[int] = None
    # Unsupervised learning params
    fixmatch: Optional[bool] = False
    unsup_weight: Optional[float] = 0.5
    pseudo_threshold: Optional[float] = None
    contrastive_temp: float = 0.1  # temperature for supervised contrastive stage
    # Per-class weights to use for the loss function
    # Underrepresented classes should have greater weight than common classes
    loss_weights: Optional[Tuple[float, ...]] = None
    # Number of outputs in final fully connected layer
    num_classes: int = 37

    # Scheduler
    use_scheduler: Optional[bool] = False
    scheduler_type: Optional[Literal["plateau"]] = None

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
            if augmentation.is_transform(obj):
                return augmentation.serialize(obj)
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
    training_pre_loop_elapsed: float

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
        if trainer.unlabelled_train_loader is not None:
            return max_num_epochs * (len(trainer.labelled_train_loader) + len(trainer.unlabelled_train_loader))
        else:
            return max_num_epochs * len(trainer.labelled_train_loader)

    def should_stop(self, trainer: 'Trainer') -> bool:
        return trainer.epoch >= trainer.params.n_epochs


class FinishedEpochs(StopCondition):
    def __init__(self, epoch_count):
        self.epoch_count = epoch_count  # Number of epochs to finish before stopping
        self.epoch_at_start_of_session = None

    def remaining_steps(self, trainer: 'Trainer') -> int:
        self.epoch_at_start_of_session = trainer.epoch
        remaining_epochs = self.epoch_count - (trainer.epoch - self.epoch_at_start_of_session)
        if trainer.unlabelled_train_loader is not None:
            return remaining_epochs * (len(trainer.labelled_train_loader) + len(trainer.unlabelled_train_loader))
        else:
            return remaining_epochs * len(trainer.labelled_train_loader)

    def should_stop(self, trainer: 'Trainer') -> bool:
        return trainer.epoch >= self.epoch_at_start_of_session + self.epoch_count


class Trainer:
    def __init__(self, params: TrainParams, determinism: Determinism = None, verbose=True):
        if determinism is not None:
            # If you want consecutive trainings to yield identical results, you need to make sure to do this
            determinism.sow(params.seed)
        params = params.copy()
        self.determinism = determinism
        self.verbose = verbose
        self.device = self._make_device()
        self.base_transform = self.make_base_transform(params)
        self.training_transform = self.make_training_transform(params)
        self.fixmatch_transform = self.make_fixmatch_transform(params)
        self.model = self._make_model(params, self.device)
        self.optimizer = self._make_optimizer(params, self.model)
        self.epoch_to_unfreezing = make_unfreezings(params.unfreezing_epochs, self.model)
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
        self.masks: dict[str, torch.Tensor] = {}
        self.scheduler = None
        if getattr(params, "scheduler_type", None) == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',  # Maximize validation accuracy
                factor=0.5,  # Reduce LR by half
                patience=2,  # Wait 2 bad epochs before reducing LR
                verbose=True,
                min_lr=1e-6  # Don't reduce below this
            )

    @classmethod
    def _make_model(cls, params: TrainParams, device):
        weights, model_fn = augmentation.arch_dict[params.architecture]
        model = model_fn(weights=weights)

        if params.freeze_layers:
            for param in model.parameters():
                param.requires_grad = False  # Freeze all layers
        for param in model.fc.parameters():
            param.requires_grad = True  # Unfreeze final layer

        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=params.augmentation.dropout_rate if params.augmentation else 0.0),
            nn.Linear(num_features, params.num_classes)
        )

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
    def make_base_transform(cls, params: TrainParams):
        """ Should be applied to validation set and test set """
        return augmentation.make_base_transform(params.architecture)

    @classmethod
    def make_training_transform(cls, params: TrainParams):
        """ Should be applied to training set only """
        base_tf = cls.make_base_transform(params)
        if not params.augmentation.enabled:
            return base_tf
        return params.augmentation.transform

    @classmethod
    def make_fixmatch_transform(cls, params: TrainParams):
        if not params.fixmatch:
            return None
        return augmentation.create_fixmatch_transform(params.architecture)

    def gpu_acceleration_enabled(self):
        return self.device.type == 'cuda'

    def maybe_unfreeze(self, epoch: int):
        if epoch not in self.epoch_to_unfreezing.keys():
            return
        layer_label, layer = self.epoch_to_unfreezing[epoch]
        for param in layer.parameters():
            param.requires_grad = True
        # print(f"Unfroze {layer_label} at epoch {epoch}")
        # Recreate optimizer after unfreezing more layers
        self.optimizer = self._make_optimizer(self.params, self.model)

    def maybe_semisupervised_learning(self, model, criterion, running_loss, pb_update_steps):
        if self.unlabelled_train_loader is None:
            return running_loss, pb_update_steps

        unsup_weight = self.params.unsup_weight
        pseudo_threshold = self.params.pseudo_threshold

        for (weak_aug, strong_aug), _ in self.unlabelled_train_loader:
            weak_aug = weak_aug.to(self.device)
            strong_aug = strong_aug.to(self.device)

            # Generate pseudo-labels using weakly augmented images
            with torch.no_grad():
                logits_u = model(weak_aug)
                probs_u = F.softmax(logits_u, dim=1)
                pseudo_labels = probs_u.argmax(dim=1)

            if pseudo_threshold is not None:
                confidence, _ = probs_u.max(dim=1)
                mask = confidence.ge(pseudo_threshold)

                if mask.any():
                    # Only use high-confidence predictions
                    strong_aug_filtered = strong_aug[mask]
                    pseudo_labels_filtered = pseudo_labels[mask]

                    # Train on strongly augmented images using pseudo-labels
                    self.optimizer.zero_grad()
                    outputs = model(strong_aug_filtered)
                    unsup_loss = criterion(outputs, pseudo_labels_filtered)

                    # Apply weight during backpropagation
                    weighted_loss = unsup_weight * unsup_loss
                    weighted_loss.backward()
                    self.optimizer.step()

                    running_loss += unsup_weight * unsup_loss.item()
            else:
                # Use all pseudo-labels without thresholding
                self.optimizer.zero_grad()
                outputs = model(strong_aug)
                unsup_loss = criterion(outputs, pseudo_labels)

                # Apply weight during backpropagation
                weighted_loss = unsup_weight * unsup_loss
                weighted_loss.backward()
                self.optimizer.step()

                running_loss += unsup_weight * unsup_loss.item()

            self.update_step += 1
            if self.verbose:
                pb_update_steps.update(1)

        return running_loss, pb_update_steps

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

    def maybe_unfreeze_last_layers(self):
        if self.params.unfreeze_last_l_blocks is not None and self.params.mft.enabled:
            raise NotImplementedError
        maybe_unfreeze_last_layers(self.params.unfreeze_last_l_blocks, self.model)

    def load_dataset(self, labelled_train_loader, unlabelled_train_loader, val_loader):
        self.labelled_train_loader = labelled_train_loader
        self.unlabelled_train_loader = unlabelled_train_loader
        self.val_loader = val_loader

    def train(self, stop_condition: StopCondition) -> TrainingResult:
        training_start = time.perf_counter()
        if self.labelled_train_loader is None or self.val_loader is None:
            raise ValueError("Must call Trainer.load(...) before training")
        if self.params.loss_weights is not None:
            loss_weights_tensor = torch.tensor(self.params.loss_weights, dtype=torch.float32, device=self.device)
        else:
            loss_weights_tensor = None
        criterion = nn.CrossEntropyLoss(weight=loss_weights_tensor)

        max_num_epochs = self.params.n_epochs
        num_epochs = max_num_epochs
        self.model.train()

        self.maybe_unfreeze_last_layers()
        self.maybe_mask_fine_tune()

        remaining_steps = stop_condition.remaining_steps(self)
        pb_update_steps = None
        if self.verbose:
            pb_update_steps = tqdm(range(1, remaining_steps + 1), desc="Update step", leave=True)  # Progress bar
        training_pre_loop_elapsed = time.perf_counter() - training_start

        while not stop_condition.should_stop(self):
            self.epoch += 1
            running_loss = 0.0
            correct = 0
            total = 0

            self.maybe_unfreeze(self.epoch)
            self.maybe_mask_fine_tune()

            # psuedo-labelling
            running_loss, pb_update_steps = self.maybe_semisupervised_learning(self.model, criterion, running_loss,
                                                                               pb_update_steps)

            for inputs, labels in self.labelled_train_loader:
                if self.verbose:
                    pb_update_steps.update(1)  # Move the progress bar by 1
                self.update_step += 1
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs, loss = backward_pass(self, inputs, labels, criterion)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_acc = correct / total

            should_record_metrics = self.should_record_metrics()
            if should_record_metrics:
                val_acc = evaluate(self.model, self.device, self.val_loader)
                val_acc_str = f", Val Acc: {100 * val_acc:.2f}%"
            else:
                val_acc = None
                val_acc_str = ""
            self.training_losses.append(running_loss / len(self.labelled_train_loader))
            self.training_accuracies.append(train_acc)
            if should_record_metrics:
                self.recorded_update_steps.append(self.update_step)
                self.validation_accuracies.append(val_acc)
                if self.scheduler is not None and val_acc is not None:
                    self.scheduler.step(val_acc)

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
            training_pre_loop_elapsed=training_pre_loop_elapsed,
        )

    def maybe_mask_fine_tune(self):
        if not self.params.mft.enabled:
            return
        suppress_weights_only_warning()
        if not self.masks:
            # Parameter selection in GPS
            self.masks = compute_gradient_masks(
                self.model, self.labelled_train_loader, self.device, self.params.mft.k
            )
        # Determine which modules are currently unfrozen (prefixes)
        allowed = set(
            name.split('.')[0]
            for name, param in self.model.named_parameters()
            if param.requires_grad
        )
        # Apply masking only to those
        apply_masks_and_freeze(self.model, self.masks, allowed)

    def save(self, dataset_params: DatasetParams):
        path = self.make_trainer_path(dataset_params, self.params)
        self.save_checkpoint(path)

    @classmethod
    def load(cls, dataset_params: DatasetParams, training_params: TrainParams, determinism: Determinism = None):
        path = cls.make_trainer_path(dataset_params, training_params)
        trainer = Trainer(training_params, determinism)
        trainer.load_checkpoint(path)
        return trainer

    @classmethod
    def make_trainer_path(cls, dataset_params: DatasetParams, training_params: TrainParams):
        # DatasetParams and TrainingParams should together be able to uniquely identify a Trainer
        key = hashlib.md5(f"{dataset_params}{training_params}".encode()).hexdigest()
        return f"runs/checkpoints/{key}.pth"

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save({
            'epoch': self.epoch,
            'update_step': self.update_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_losses': self.training_losses,
            'training_accuracies': self.training_accuracies,
            'validation_accuracies': self.validation_accuracies,
        }, path)
        if self.verbose:
            print(f"[Trainer] Saved checkpoint to {path}")

    def load_checkpoint(self, path: str, map_location=None):
        checkpoint = torch.load(path, map_location=map_location or self.device)
        self.epoch = checkpoint.get('epoch', 0)
        self.update_step = checkpoint.get('update_step', 0)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # 3) replay the freeze/unfreeze schedule up to self.epoch
        #    a) freeze everything if you used freeze_layers=True
        if self.params.freeze_layers:
            for p in self.model.parameters():
                p.requires_grad = False

        #    b) always unfreeze the final FC layer
        for p in self.model.fc.parameters():
            p.requires_grad = True

        #    c) unfreeze any of layer3/layer4 that you’d un-frozen during training
        #       (your epoch_to_unfreezing maps epoch→(label, layer_module))
        for unfreeze_epoch, (label, layer_mod) in self.epoch_to_unfreezing.items():
            if unfreeze_epoch <= self.epoch:
                for p in layer_mod.parameters():
                    p.requires_grad = True

        # 4) now rebuild your optimizer so its param-groups line up
        self.optimizer = self._make_optimizer(self.params, self.model)

        # 5) load the saved optimizer state (now the group sizes match)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.training_losses = list(checkpoint.get('training_losses', []))
        self.training_accuracies = list(checkpoint.get('training_accuracies', []))
        self.validation_accuracies = list(checkpoint.get('validation_accuracies', []))
        if self.verbose:
            print(f"[Trainer] Loaded checkpoint from {path} (epoch {self.epoch})")


def terminate_workers(labelled_train_loader, unlabelled_train_loader, val_loader):
    for loader in (labelled_train_loader, unlabelled_train_loader, val_loader):
        it = getattr(loader, "_iterator", None)
        if it is not None:
            it._shutdown_workers()
            del loader._iterator
