import copy
import hashlib
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Literal, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm

import augmentation
from augmentation import make_augmented_transform
from datasets import DatasetParams
from determinism import Determinism
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
    pseudo_threshold: Optional[float] = None
    # Masked fine-tuning
    masked_finetune: bool = False  # whether to run GPS-style masked fine-tuning
    mask_K: int = 1  # number of weights to update per neuron
    contrastive_temp: float = 0.1  # temperature for supervised contrastive stage

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
    num_classes = 37

    def __init__(self, params: TrainParams, determinism: Determinism = None, verbose=True):
        if determinism is not None:
            determinism.sow(params.seed)  # If you want consecutive trainings to yield identical results, you need to make sure to do this
        params = params.copy()
        self.determinism = determinism
        self.verbose = verbose
        self.device = self._make_device()
        self.base_transform = self.make_base_transform(params)
        self.training_transform = self.make_training_transform(params)
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
        self._masks: dict[str, torch.Tensor] = {}

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
    def make_base_transform(cls, params: TrainParams):
        """ Should be applied to validation set and test set """
        return augmentation.make_base_transform(params.architecture)

    @classmethod
    def make_training_transform(cls, params: TrainParams):
        """ Should be applied to training set only """
        base_tf = cls.make_base_transform(params)
        if params.data_augmentation != "true":
            return base_tf
        return make_augmented_transform(base_tf)

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

    def maybe_semisupervised_learning(self, model, criterion, running_loss, pb_update_steps):
        if self.unlabelled_train_loader is None:
            return running_loss, pb_update_steps

        unsup_weight = self.params.unsup_weight
        pseudo_threshold = self.params.pseudo_threshold
        for batch_u in self.unlabelled_train_loader:
            if isinstance(batch_u, (list, tuple)):
                x_u = batch_u[0]
            else:
                x_u = batch_u

            x_u = x_u.to(self.device)

            with torch.no_grad():
                logits_u = model(x_u)
                probs_u = F.softmax(logits_u, dim=1)
                pseudo = probs_u.argmax(dim=1)

            if pseudo_threshold is not None:
                confidence, _ = probs_u.max(dim=1)
                mask = confidence.ge(pseudo_threshold)

                if mask.any():
                    inputs_u, labels_u = x_u[mask], pseudo[mask]
                    self.optimizer.zero_grad()
                    _, unsup_loss = backward_pass(self, inputs_u, labels_u, criterion)
                    running_loss += unsup_weight * unsup_loss.item()

            else:
                # train on all pseudo-labels with no threshold
                self.optimizer.zero_grad()
                _, unsup_loss = backward_pass(self, x_u, pseudo, criterion)
                running_loss += unsup_weight * unsup_loss.item()

            self.update_step += 1
            if self.verbose:
                pb_update_steps.update(1)  # Move the progress bar by 1

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

    def load_dataset(self, labelled_train_loader, unlabelled_train_loader, val_loader):
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
        self.maybe_mask_fine_tune()

        remaining_steps = stop_condition.remaining_steps(self)
        pb_update_steps = None
        if self.verbose:
            pb_update_steps = tqdm(range(1, remaining_steps + 1), desc="Update step", leave=True)  # Progress bar

        while not stop_condition.should_stop(self):
            self.epoch += 1
            running_loss = 0.0
            correct = 0
            total = 0

            self.maybe_unfreeze(self.epoch)

            # psuedo-labelling
            running_loss, pb_update_steps = self.maybe_semisupervised_learning(model, criterion, running_loss,
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

    def maybe_mask_fine_tune(self):
        # if masked fine-tuning is requested, run the mask computation
        if self.params.masked_finetune:
            self._compute_masks()
            # attach gradient hooks so only masked entries get nonzero grads
            for name, W in self.model.named_parameters():
                mask = self._masks.get(name)
                if mask is not None:
                    W.requires_grad_(True)

                    def hook(g, mask=mask):
                        return g * mask.to(g.device)

                    W.register_hook(hook)
            # make sure the classifier head is trainable again
            for p in self.model.fc.parameters():
                p.requires_grad = True
            # rebuild optimizer so it includes the now-trainable fc and masked backbone weights
            self.optimizer = self._make_optimizer(self.params, self.model)

    def _compute_masks(self):
        """
        Stage 1: run one forward/backward pass with a tiny projection head + SupCon loss,
        then for each weight tensor pick the top-K gradients per output neuron.
        """
        # freeze current classifier head
        for p in self.model.fc.parameters():
            p.requires_grad = False
        # swap out fc → Identity so we get features
        orig_fc = self.model.fc
        self.model.fc = nn.Identity()

        # build a small projection head on top of features
        d = orig_fc.in_features
        proj_head = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
        ).to(self.device)

        # one gradient pass
        self.model.train()
        proj_head.train()
        inputs, labels = next(iter(self.labelled_train_loader))
        inputs, labels = inputs.to(self.device), labels.to(self.device)

        # compute SupCon loss (you'll need a SupCon implementation)
        features = self.model(inputs)
        z = proj_head(features)
        from masked_fine_tuning import supervised_contrastive_loss
        loss = supervised_contrastive_loss(z, labels, self.params.contrastive_temp)
        loss.backward()

        # build masks: for each weight tensor select top-K grads per output neuron
        for name, W in self.model.named_parameters():
            if W.grad is None: continue
            G = W.grad.abs()
            mask = torch.zeros_like(G)
            K = self.params.mask_K
            # assume W shape [in_dim, out_dim]
            topk = torch.topk(G, k=K, dim=0).indices  # shape [K, out_dim]
            mask[topk, torch.arange(G.size(1))] = 1
            self._masks[name] = mask

        # restore fc, clear grads
        self.model.fc = orig_fc
        # re-enable the classifier head for the upcoming cross-entropy stage
        for p in self.model.fc.parameters():
            p.requires_grad = True

        proj_head.zero_grad()
        for p in self.model.parameters():
            p.grad = None

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
