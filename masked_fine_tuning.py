import torch
import torch.nn.functional as F


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor,
                                temperature: float = 0.07) -> torch.Tensor:
    """
    Supervised Contrastive Loss as described in:
    Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.

    Args:
        features: Tensor of shape [batch_size, feature_dim].
                  Features should be the output of a projection head.
        labels:   Tensor of shape [batch_size] with integer class labels.
        temperature: Scalar temperature for scaling similarities.

    Returns:
        A scalar tensor representing the loss.
    """
    device = features.device
    batch_size = features.size(0)

    # Normalize feature vectors
    features_normalized = F.normalize(features, p=2, dim=1)

    # Compute pairwise similarity matrix
    similarity_matrix = torch.matmul(features_normalized, features_normalized.T) / temperature

    # For numerical stability, subtract max logit per row
    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()

    # Build mask where mask[i, j] = 1 if labels[i] == labels[j], else 0
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)
    # Zero out self-contrast cases
    mask.fill_diagonal_(0)

    # Compute exp logits, zeroing out self-similarities
    exp_logits = torch.exp(logits) * (1 - torch.eye(batch_size, device=device))

    # Log probabilities
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # Compute mean log-probability of positive pairs
    # Avoid division by zero by adding a small epsilon
    eps = 1e-12
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + eps)

    # Loss is the average negative log-likelihood over positives
    loss = -mean_log_prob_pos.mean()
    return loss
