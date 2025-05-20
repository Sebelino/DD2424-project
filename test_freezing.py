import json

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import OxfordIIITPet

from freezing import compute_gradient_masks_no_cache


def test_compute_gradient_masks():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = OxfordIIITPet(root="./data", split="trainval", target_types="category", download=True, transform=transform)
    dl = DataLoader(ds, batch_size=2, shuffle=False)

    # Prepare a ResNet-50 model
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 37)
    model = model.to(device)

    # Run mask computation with a small K
    K = 5
    masks, summary = compute_gradient_masks_no_cache(model, dl, device, K)

    print(json.dumps(summary))
    selected_param_count = sum(a for a, b, c in summary.values())
    total_param_count = sum(b for a, b, c in summary.values())
    assert summary["conv1.weight"] == (K * 64, 9408, (64, 3, 7, 7))
    assert summary["bn1.weight"] == (64, 64, (64,))
    assert summary["fc.bias"] == (0, 37, (37,))
    assert selected_param_count == 182265
    assert total_param_count == 23583845

    # Assertions:
    for name, param in model.named_parameters():
        assert name in masks, f"Missing mask for parameter: {name}"
        mask = masks[name]
        # 1. Shape matches
        assert mask.shape == param.shape, f"Mask shape {mask.shape} != param shape {param.shape} for {name}"

        # 2. Binary values
        unique_vals = torch.unique(mask)
        for val in unique_vals:
            assert val.item() in (0.0, 1.0), f"Mask for {name} contains non-binary value {val}"

        # 3. Row-sum constraint: for weight tensors with >1 input per neuron,
        #    sum over inputs should equal min(K, number of inputs), per neuron.
        #    Ignore 1-D tensors (biases, batch-norm), which collapse to rows of length 1.
        if mask.ndim >= 2:
            flat = mask.view(mask.size(0), -1)
            rowsums = flat.sum(dim=1)
            expected = min(K, flat.size(1))
            assert torch.all(rowsums == expected), (
                f"Row sums for {name} = {rowsums.tolist()}, expected {expected}"
            )
