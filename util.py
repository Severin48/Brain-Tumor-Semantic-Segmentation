import torch

def _binarise(preds: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    """
    Turn a (B,1,H,W) tensor of logits **or** probabilities into a
    hard {0,1} mask.
    If `preds` is floating-point we assume logits and apply sigmoid.
    """
    if preds.dtype.is_floating_point:
        preds = torch.sigmoid(preds)
    return (preds > thr).float()


def dice_coefficient(preds: torch.Tensor,
                     targets: torch.Tensor,
                     eps: float = 1e-6,
                     thr: float = 0.5) -> float:
    """
    Classic Sörensen-Dice averaged over the batch

    Args
    ----
    preds   : raw logits **or** probabilities – shape (B,1,H,W)
    targets : binary ground-truth masks     – shape (B,1,H,W)

    Returns
    -------
    Scalar Dice in [0,1].
    """
    preds_bin   = _binarise(preds, thr)
    targets_bin = targets.float()

    # Flatten (B,1,H,W) -> (B, H*W)  and compute per-image Dice
    preds_flat   = preds_bin.reshape(preds_bin.size(0), -1)
    targets_flat = targets_bin.reshape(targets_bin.size(0), -1)

    intersection = (preds_flat * targets_flat).sum(1)
    denom        = preds_flat.sum(1) + targets_flat.sum(1)

    dice = (2 * intersection + eps) / (denom + eps)
    return dice.mean().item()



def iou_score(preds: torch.Tensor,
              targets: torch.Tensor,
              eps: float = 1e-6,
              thr: float = 0.5) -> float:
    """
    Intersection-over-Union averaged over the batch
    """
    preds_bin   = _binarise(preds, thr)
    targets_bin = targets.float()

    preds_flat   = preds_bin.reshape(preds_bin.size(0), -1)
    targets_flat = targets_bin.reshape(targets_bin.size(0), -1)

    intersection = (preds_flat * targets_flat).sum(1)
    union        = preds_flat.sum(1) + targets_flat.sum(1) - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()
