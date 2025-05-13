import torch
from torchmetrics.segmentation import DiceScore
from torchmetrics.classification import BinaryJaccardIndex

_dice_metric = DiceScore(
    num_classes=1,
    include_background=False,
    average='micro',
    input_format='one-hot'
)

_iou_metric = BinaryJaccardIndex(
    threshold=0.5,
    ignore_index=None,
    validate_args=False,
    zero_division=0
)


def dice_coefficient(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    preds: raw logits or probabilities, shape (B, 1, H, W)
    targets: binary masks {0,1}, shape (B, 1, H, W)
    Returns the Dice coefficient (scalar float).
    """
    device = preds.device
    # Convert logits → probabilities → binary
    probs = torch.sigmoid(preds) if preds.dtype.is_floating_point else preds
    preds_bin = (probs > 0.5).int().cpu()
    targets_bin = targets.int().cpu()

    # DiceScore expects boolean one-hot of shape (B, C, H, W)
    dice_val = _dice_metric(preds_bin.bool(), targets_bin.bool())
    return dice_val.item()


def iou_score(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    preds: raw logits or probabilities, shape (B, 1, H, W)
    targets: binary masks {0,1}, shape (B, 1, H, W)
    Returns the Intersection over Union (scalar float).
    """
    # BinaryJaccardIndex handles logits internally
    preds_cpu = preds.cpu()
    targets_cpu = targets.int().cpu()

    iou_val = _iou_metric(preds_cpu, targets_cpu)
    return iou_val.item()
