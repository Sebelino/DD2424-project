from dataclasses import dataclass

import math
import numpy as np
import torch
import torch.nn as nn
from joblib import Memory

USE_CACHE = True

if USE_CACHE:
    memory = Memory("./runs/joblib_cache", verbose=0)
else:
    memory = Memory(location=None, verbose=0)


def make_unfreezings(unfreezing_epochs, model):
    # Define layers to gradually unfreeze from last to first
    layer_labels = ["layer4", "layer3", "layer2", "layer1", "conv1"]
    layer_modules = [model.layer4, model.layer3, model.layer2, model.layer1, model.conv1]

    if unfreezing_epochs in {None, ()}:
        return {}

    unfreezing_epochs = sorted(unfreezing_epochs)
    if len(unfreezing_epochs) > len(layer_modules):
        raise ValueError(f"Cannot unfreeze more than {len(layer_modules)} layers, got {len(unfreezing_epochs)}.")

    return {
        unfreezing_epochs[i]: (layer_labels[i], layer_modules[i])
        for i in range(len(unfreezing_epochs))
    }


def maybe_unfreeze_last_layers(l, model: nn.Module):
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


@dataclass
class MaskedFineTuningParams:
    enabled: bool
    k: int  # Number of weights to update per neuron
    impl: str

    def __reduce__(self):
        return MaskedFineTuningParams, (self.enabled, self.k, self.impl)


@memory.cache(ignore=["model", "dataloader", "device"])
def compute_gradient_masks(model, dataloader, device, k):
    return compute_gradient_masks_no_cache(model, dataloader, device, k)


def compute_gradient_masks_no_cache(model, dataloader, device, k):
    model.train()
    criterion = nn.CrossEntropyLoss()
    grad_sums = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_sums[name] += param.grad.abs()
    summary = {}
    masks = {}
    for name, grad in grad_sums.items():
        flat = grad.view(grad.size(0), -1)
        if "norm" in name or "bn" in name or "pos_embed" in name or "cls_token" in name:
            mask_flat = torch.ones_like(flat)
        elif "head" in name or "bias" in name or "gamma" in name:
            mask_flat = torch.zeros_like(flat)
        else:
            max_inputs = flat.size(1)
            top_k = min(k, max_inputs)
            if top_k < 1:
                masks[name] = torch.zeros_like(grad)
                continue
            _, idx = torch.topk(flat, top_k, dim=1)
            mask_flat = torch.zeros_like(flat)
            mask_flat.scatter_(1, idx, 1.0)
        masks[name] = mask_flat.view_as(grad)
        summary[name] = (int(mask_flat.sum()), math.prod(list(mask_flat.size())), grad.shape)
    torch.save(masks, f"masks_k={k}.pt")
    return masks, summary


def prune(model, dataloader, device, k):
    # Populate model.param.grad
    model.train()
    model = model.to(device)
    model.zero_grad()
    imgs, labels = next(iter(dataloader))
    imgs, labels = imgs.to(device), labels.to(device)
    outputs = model(imgs)
    loss = torch.nn.functional.cross_entropy(outputs, labels)
    loss.backward()

    masks, summary = prune_by_percentile_gradient_perCell(model, k)
    torch.save(masks, f"masks_k={k}.pt")
    return masks, summary


def prune_by_percentile_gradient_perCell(model, k):
    summary = {}
    new_masks = {}

    for name, param in model.named_parameters():
        if "norm" in name or "bn" in name or "pos_embed" in name or "cls_token" in name:
            mask_np = np.ones_like(param.data.cpu().numpy())
        elif "head" in name or "bias" in name or "gamma" in name:
            mask_np = np.zeros_like(param.data.cpu().numpy())
        else:
            if param.grad is None:
                mask_np = np.zeros_like(param.data.cpu().numpy())
            else:
                grad_np = param.grad.data.abs().cpu().numpy()
                if grad_np.ndim == 4:
                    B, C, H, W = grad_np.shape
                    flat = grad_np.reshape(B, -1)
                elif grad_np.ndim == 2:
                    flat = grad_np
                elif grad_np.ndim == 1:
                    flat = grad_np.reshape(-1, 1)
                else:
                    flat = grad_np.reshape(grad_np.shape[0], -1)

                mask_flat = np.zeros_like(flat, dtype=np.float32)
                idx = np.argsort(flat, axis=1)[:, -k:]
                for i in range(mask_flat.shape[0]):
                    mask_flat[i, idx[i]] = 1.0

                mask_np = mask_flat.reshape(param.data.cpu().numpy().shape)

        total = mask_np.size
        kept = int(mask_np.sum())
        summary[name] = (kept, total, param.data.cpu().numpy().shape)
        new_masks[name] = torch.from_numpy(mask_np).to(param.device)
    return new_masks, summary


def apply_masks_and_freeze(model, masks=None, allowed_prefixes=None):
    for name, param in model.named_parameters():
        prefix = name.split('.')[0]
        if masks is None:
            param.requires_grad = True
            continue
        if allowed_prefixes is not None and prefix not in allowed_prefixes:
            param.requires_grad = False
            continue
        mask = masks.get(name)
        if mask is not None:
            param.requires_grad = True
            m = mask.to(param.device)

            def make_hook(mask_tensor):
                def hook(grad):
                    return grad * mask_tensor

                return hook

            param.register_hook(make_hook(m))
        else:
            param.requires_grad = False
