import torch
import torch.nn as nn
import torch.nn.functional as F

class MulticlassCrossEntropyDiceLoss(nn.Module):
    """Ignore-safe per-sample cross-entropy plus soft Dice loss.

    The return shape follows CeDiRNet's criterion contract: total loss first,
    followed by component losses, with every tensor retaining the batch axis.
    """

    def __init__(
        self,
        class_weights,
        ignore_index=255,
        sample_key="semantic_segmentation",
        cross_entropy_weight=1.0,
        dice_weight=1.0,
        epsilon=1.0e-6,
        task_name="semantic_segmentation",
        **kwargs,
    ):
        super().__init__()
        weights = torch.as_tensor(class_weights, dtype=torch.float32)
        if weights.ndim != 1 or weights.numel() < 2:
            raise ValueError("class_weights must contain at least two classes")
        if torch.any(weights <= 0) or not torch.isfinite(weights).all():
            raise ValueError("class_weights must be finite and positive")
        self.register_buffer("class_weights", weights)
        self.ignore_index = int(ignore_index)
        self.sample_key = sample_key
        self.cross_entropy_weight = float(cross_entropy_weight)
        self.dice_weight = float(dice_weight)
        self.epsilon = float(epsilon)
        self.task_name = task_name

    def forward(self, logits, sample, **kwargs):
        target = sample[self.sample_key]
        if target.ndim == 4 and target.shape[1] == 1:
            target = target[:, 0]
        if target.ndim != 3:
            raise ValueError("semantic target must have shape BxHxW or Bx1xHxW")
        if logits.ndim != 4 or logits.shape[0] != target.shape[0]:
            raise ValueError("logits and target batch dimensions do not match")
        if logits.shape[1] != self.class_weights.numel():
            raise ValueError("logit channels do not match class_weights")
        if logits.shape[-2:] != target.shape[-2:]:
            raise ValueError("logits and target spatial dimensions do not match")

        target = target.long()
        valid = target != self.ignore_index
        safe_target = torch.where(valid, target, torch.zeros_like(target))
        if torch.any((safe_target < 0) | (safe_target >= logits.shape[1])):
            raise ValueError("semantic target contains an out-of-range class id")

        batch_size = logits.shape[0]
        if not torch.any(valid):
            zero = logits.sum() * 0.0
            zeros = zero.expand(batch_size)
            return zeros, zeros, zeros

        cross_entropy_scalar = F.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
        )
        probabilities = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(safe_target, num_classes=logits.shape[1])
        one_hot = one_hot.permute(0, 3, 1, 2).to(logits.dtype)
        valid_channels = valid.unsqueeze(1).to(logits.dtype)
        probabilities = probabilities * valid_channels
        one_hot = one_hot * valid_channels
        reduce_dims = (0, 2, 3)
        intersection = (probabilities * one_hot).sum(reduce_dims)
        denominator = probabilities.sum(reduce_dims) + one_hot.sum(reduce_dims)
        present = one_hot.sum(reduce_dims) > 0
        dice = (2.0 * intersection + self.epsilon) / (
            denominator + self.epsilon
        )
        dice_loss_scalar = (
            (1.0 - dice[present]).mean()
            if torch.any(present)
            else logits.sum() * 0.0
        )
        total_scalar = (
            self.cross_entropy_weight * cross_entropy_scalar
            + self.dice_weight * dice_loss_scalar
        )
        return (
            total_scalar.expand(batch_size),
            cross_entropy_scalar.expand(batch_size),
            dice_loss_scalar.expand(batch_size),
        )

    def get_loss_dict(self, loss_tensor):
        _, cross_entropy, dice_loss = loss_tensor[:3]
        task_total = (
            self.cross_entropy_weight * cross_entropy
            + self.dice_weight * dice_loss
        )
        return {
            "loss": task_total,
            "semantic_cross_entropy": cross_entropy,
            "semantic_dice": dice_loss,
            "losses_tasks": {self.task_name: task_total.sum()},
            "losses_groups": {self.task_name: task_total.sum()},
        }
