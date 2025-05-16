import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
from util import dice_coefficient, iou_score
import torch.nn.functional as F


def dice_coef_loss(inputs, target):
    smooth = 1.0
    intersection = 2.0 * ((target * inputs).sum()) + smooth
    union = target.sum() + inputs.sum() + smooth

    return 1 - (intersection / union)

def bce_dice_loss(inputs, target):
    dicescore = dice_coef_loss(torch.sigmoid(inputs), target)
    loss_fn = nn.BCEWithLogitsLoss()
    bce_loss = loss_fn(inputs, target)
    return bce_loss + dicescore

def train(model,
          train_loader,
          val_loader,
          device,
          epochs=10,
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
          'history': {'train_loss': [...], 'val_dice': [...], 'val_iou': [...]},
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

    history = {"train_loss": [], "val_dice": [], "val_iou": []}
    best_dice = -1.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)  # outputs are logits (before sigmoid)

            # Calculate loss combo (BCE + Dice)
            loss = bce_dice_loss(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}")
        # Evaluation
        model.eval()
        val_dices, val_ious = [], []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_dices.append(dice_coefficient(outputs, masks))
                val_ious.append(iou_score(outputs, masks))

        val_dice = sum(val_dices) / len(val_dices)
        val_iou = sum(val_ious) / len(val_ious)

        history["train_loss"].append(train_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)

        # Keep best
        if val_dice > best_dice:
            best_dice = val_dice
            best_weights = model.state_dict()

        loop.set_postfix(train_loss=f"{train_loss:.4f}", val_dice=f"{val_dice:.4f}", val_iou=f"{val_iou:.4f}")

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
