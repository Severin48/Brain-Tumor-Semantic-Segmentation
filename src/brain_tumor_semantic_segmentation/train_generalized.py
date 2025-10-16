from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
from typing import Type, Optional, Dict, Any, Callable
import torch.nn.functional as F
from src.brain_tumor_semantic_segmentation.util import dice_coefficient, iou_score

def dice_loss(inputs, target, eps=1e-6):
    inputs = torch.sigmoid(inputs)
    intersection = 2.0 * ((target * inputs).sum()) + eps
    union = target.sum() + inputs.sum() + eps

    return 1 - (intersection / union)

def bce_loss(inputs, target):
    loss_fn = nn.BCEWithLogitsLoss()
    bce_loss = loss_fn(inputs, target)
    return bce_loss

def bce_dice_loss(inputs, target, smoothing_factor=0.9):
    bce_score = bce_loss(inputs, target)
    dice_score = dice_loss(inputs, target)

    # dynamic weighting based on scores + smoothing to prevent low training
    bce_weight = smoothing_factor * bce_score.item() / (bce_score.item() + dice_score.item() + 1e-8) + (1 - smoothing_factor)
    dice_weight = smoothing_factor * dice_score.item() / (bce_score.item() + dice_score.item() + 1e-8) + (1 - smoothing_factor)
    #print(f"bce_weight: {bce_weight:.4f}, dice_weight: {dice_weight:.4f}")
    return bce_weight * bce_score + dice_weight * dice_score

def focal_loss(inputs, target, alpha=0.8, gamma=2):
    inputs = torch.sigmoid(inputs)
    bce = F.binary_cross_entropy(inputs, target, reduction='mean')
    bce_exp = torch.exp(-bce)
    focal_loss = alpha * (1-bce_exp)**gamma * bce
    return focal_loss

def tversky_loss(inputs, target, alpha=0.5, beta=0.5, eps=1e-6):
    inputs = torch.sigmoid(inputs)
    
    # Flatten the tensors
    inputs = inputs.view(-1)
    target = target.view(-1)
    
    TP = (inputs * target).sum()
    FP = ((1-target) * inputs).sum()
    FN = (target * (1-inputs)).sum()
    
    tversky = (TP + eps) / (TP + alpha*FP + beta*FN + eps)  
    return 1 - tversky

def train(model,
          train_loader,
          val_loader,
          device,
          epochs: int = 10,
          lr: float = 1e-3,
          lr_sched_cls: Optional[Type[torch.optim.lr_scheduler._LRScheduler]] = None,
          lr_sched_kwargs: Optional[Dict[str, Any]] = None,
          optimizer_class=optim.Adam,
          loss_fn: Callable = bce_loss, # Callable means it can be a function (bce_loss) or a class (nn.BCEWithLogitsLoss)
          task: str = "segmentation",  # "segmentation" or "classification"
          save_dir: str = "checkpoints",
          save_name: str | None = None):
    """
    Universal training function for segmentation and binary classification with early stopping.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    if save_name:
        fname = save_name if save_name.endswith(".pth") else f"{save_name}.pth"
    else:
        fname = datetime.now().strftime("%Y_%m_%d-%H_%M_%S.pth")
    checkpoint_path = save_path / fname

    model = model.to(device)
    optimizer = optimizer_class(model.parameters(), lr=lr)

    scheduler = None
    if lr_sched_cls is not None:
        lr_sched_kwargs = lr_sched_kwargs or {}
        scheduler = lr_sched_cls(optimizer, **lr_sched_kwargs)
    
    if isinstance(loss_fn, type):
        criterion = loss_fn()   # Instantiate if its a class (e.g., nn.BCEWithLogitsLoss)
    else:
        criterion = loss_fn     # Use directly if its a function (e.g., dice_loss)

    if task == "segmentation":
        history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []}
        best_metric = -float('inf')
    if task == "classification":
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_metric = -float('inf')

    best_weights = None

    @torch.no_grad()
    def evaluate(loader):
        total_loss = 0.0
        if task == "segmentation":
            dices, ious = [], []
            for images, masks in loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                total_loss += criterion(outputs, masks).item()
                dices.append(dice_coefficient(outputs, masks))
                ious.append(iou_score(outputs, masks))
            avg_loss = total_loss / len(loader)
            avg_dice = sum(dices) / len(dices)
            avg_iou = sum(ious) / len(ious)
            return avg_loss, avg_dice, avg_iou
        if task == "classification":
            correct, total = 0, 0
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device).long()
                outputs = model(images)
                total_loss += criterion(outputs, labels).item()
                if isinstance(criterion, nn.CrossEntropyLoss):
                    preds = torch.argmax(outputs, dim=1)
                else:
                    preds = (torch.sigmoid(outputs) > 0.5).long().squeeze(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            avg_loss = total_loss / len(loader)
            avg_acc = correct / total if total > 0 else 0.0
            return avg_loss, avg_acc

    # Initial evaluation
    model.eval()
    if task == "segmentation":
        train_loss0, train_dice0, train_iou0 = evaluate(train_loader)
        val_loss0, val_dice0, val_iou0 = evaluate(val_loader)
        history["train_loss"].append(train_loss0)
        history["val_loss"].append(val_loss0)
        history["val_dice"].append(val_dice0)
        history["val_iou"].append(val_iou0)
        best_metric = val_dice0
    if task == "classification":
        train_loss0, train_acc0 = evaluate(train_loader)
        val_loss0, val_acc0 = evaluate(val_loader)
        history["train_loss"].append(train_loss0)
        history["val_loss"].append(val_loss0)
        history["train_acc"].append(train_acc0)
        history["val_acc"].append(val_acc0)
        best_metric = val_acc0

    # Training loop
    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in loop:
            if task == "segmentation":
                images, targets = batch
                targets = targets.to(device)
            if task == "classification":
                images, targets = batch
                targets = targets.to(device).long()

            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if task == "classification":
                if isinstance(criterion, nn.CrossEntropyLoss):
                    preds = torch.argmax(outputs, dim=1)
                else:
                    preds = (torch.sigmoid(outputs) > 0.5).long().squeeze(1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

            avg_loss = running_loss / len(train_loader)
            if task == "classification":
                loop.set_postfix(train_loss=f"{avg_loss:.4f}", train_acc=f"{correct/total:.4f}")
            if task == "segmentation":
                loop.set_postfix(train_loss=f"{avg_loss:.4f}")

        print(f"Epoch {epoch}/{epochs} - Avg Train Loss: {avg_loss:.4f}")

        model.eval()

        if scheduler is not None:
            scheduler.step()
        
        if task == "segmentation":
            val_loss, val_dice, val_iou = evaluate(val_loader)
            history["train_loss"].append(avg_loss)
            history["val_loss"].append(val_loss)
            history["val_dice"].append(val_dice)
            history["val_iou"].append(val_iou)
            if val_dice > best_metric:
                best_metric = val_dice
                best_weights = model.state_dict()
            loop.set_postfix(train=f"{avg_loss:.4f}", val=f"{val_loss:.4f}", dice=f"{val_dice:.4f}", iou=f"{val_iou:.4f}")
        if task == "classification":
            val_loss, val_acc = evaluate(val_loader)
            history["train_loss"].append(avg_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(correct/total)
            history["val_acc"].append(val_acc)
            if val_acc > best_metric:
                best_metric = val_acc
                best_weights = model.state_dict()
            loop.set_postfix(train=f"{avg_loss:.4f}", val=f"{val_loss:.4f}", acc=f"{val_acc:.4f}")

    torch.save(best_weights, checkpoint_path)
    model.load_state_dict(best_weights)

    return model, {
        "model": model,
        "history": history,
        "val_loader": val_loader,
        "device": device,
        "scheduler": scheduler,
        "save_path": str(checkpoint_path)
    }
