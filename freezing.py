from dataclasses import dataclass

import torch
import torch.nn as nn
from joblib import Memory

USE_CACHE = True

if USE_CACHE:
    memory = Memory("./runs/joblib_cache", verbose=0)
else:
    memory = Memory(location=None, verbose=0)


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


def make_unfreezings(unfreezing_epochs, model):
    # Define layers to gradually unfreeze
    layer_names = [model.layer4, model.layer3]  # layer4 = last block
    if unfreezing_epochs in {None, ()}:
        return dict()
    unfreezing_epochs = sorted(unfreezing_epochs)
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


@dataclass
class MaskedFineTuningParams:
    enabled: bool
    k: int  # Number of weights to update per neuron

    def __reduce__(self):
        return MaskedFineTuningParams, (self.enabled, self.k)


@memory.cache(ignore=["model", "dataloader", "device"])
def compute_gradient_masks(model, dataloader, device, k):
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
    masks = {}
    for name, grad in grad_sums.items():
        flat = grad.view(grad.size(0), -1)
        max_inputs = flat.size(1)
        top_k = min(k, max_inputs)
        if top_k < 1:
            masks[name] = torch.zeros_like(grad)
            continue
        _, idx = torch.topk(flat, top_k, dim=1)
        mask_flat = torch.zeros_like(flat)
        mask_flat.scatter_(1, idx, 1.0)
        masks[name] = mask_flat.view_as(grad)
    torch.save(masks, f"masks_k={k}.pt")
    return masks


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
