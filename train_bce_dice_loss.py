import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
from util import dice_coefficient, iou_score
import torch.nn.functional as F

def switch_loss(inputs, target):
    if target.sum() == 0:           # no tumor present
        return bce_loss(inputs, target)
    return bce_dice_loss(inputs, target)

def dice_loss(inputs, target, eps=1e-6):
    inputs = torch.sigmoid(inputs)
    intersection = 2.0 * ((target * inputs).sum()) + eps
    union = target.sum() + inputs.sum() + eps

    return 1 - (intersection / union)

def bce_loss(inputs, target):
    loss_fn = nn.BCEWithLogitsLoss()
    bce_loss = loss_fn(inputs, target)
    return bce_loss

def bce_dice_loss(inputs, target):
    bce_score = bce_loss(inputs, target)
    dice_score = dice_loss(inputs, target)
    return bce_score + dice_score  # Check scales

def train_bce_dice_loss(loss_fn, model,
          train_loader,
          val_loader,
          device,
          epochs=50,
          lr=1e-3,
          save_dir="checkpoints",
          save_name=None):
    """
    Train a segmentation model, keeping only the best weights,
    and save checkpoint at the end with a timestamped (or custom) name.

    Returns:
        best_model: model loaded with the best validation-Dice weights
        results: {
          'model': best_model,
          'history': {'train_loss': [...], 'val_loss': [...], 'val_dice': [...], 'val_iou': [...]},
          'val_loader': val_loader,
          'device': device,
          'save_path': full path to the saved .pth
        }
    """
    # Manage checkpoint path
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if save_name:
        fname = save_name if save_name.endswith(".pth") else f"{save_name}.pth"
    else:
        fname = datetime.now().strftime("%Y_%m_%d-%H_%M_%S.pth")
    save_path = save_dir / fname

    # Training
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history   = {k: [] for k in ("train_loss", "val_loss", "val_dice", "val_iou")}
    best_dice = -1.0
    best_weights = None

    @torch.no_grad()
    def evaluate(loader):
        total_loss = 0.0
        dices, ious = [], []
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            total_loss += loss_fn(outputs, masks).item()
            dices.append(dice_coefficient(outputs, masks))
            ious.append(iou_score(outputs, masks))
        avg_loss = total_loss / len(loader)
        avg_dice = sum(dices) / len(dices)
        avg_iou  = sum(ious)  / len(ious)
        return avg_loss, avg_dice, avg_iou

    model.eval()
    train_loss0, _, _     = evaluate(train_loader)
    val_loss0,  d0,  i0   = evaluate(val_loader)

    for k, v in zip(["train_loss", "val_loss", "val_dice", "val_iou"], [train_loss0, val_loss0, d0, i0]):
        history[k].append(v)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_loss = running_loss / len(train_loader)

        # Evaluate after epoch
        model.eval()
        val_loss, val_dice, val_iou = evaluate(val_loader)

        for k, v in zip(["train_loss", "val_loss", "val_dice", "val_iou"], [train_loss, val_loss, val_dice, val_iou]):
            history[k].append(v)

        # Keep best
        if val_dice > best_dice:
            best_dice = val_dice
            best_weights = model.state_dict()

        loop.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}", dice=f"{val_dice:.4f}", iou=f"{val_iou:.4f}")


    torch.save(best_weights, save_path)

    # Reload best into model
    model.load_state_dict(best_weights)

    results = {
        "model": model,
        "history": history,
        "val_loader": val_loader,
        "device": device,
        "save_path": str(save_path)
    }
    return model, results

