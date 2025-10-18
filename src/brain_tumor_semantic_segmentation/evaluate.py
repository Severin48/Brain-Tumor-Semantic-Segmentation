from __future__ import annotations

import matplotlib.pyplot as plt
import torch
import numpy as np

from src.brain_tumor_semantic_segmentation.util import dice_coefficient, iou_score
from src.brain_tumor_semantic_segmentation.data import DATASET_MEAN, DATASET_STD


def evaluate(results: dict, num_batches: int = 1, alpha: float = 0.35) -> None:
    """Visualise training curves and qualitative segmentation results
    Plots both validation (CV) and test metrics if available
    """
    model      = results["model"]
    history    = results["history"]
    val_loader = results["val_loader"]
    device     = results["device"]

    epochs = range(len(history["train_loss"]))

    # Final scores
    dices, ious = [], []
    with torch.no_grad():
        for imgs, masks in val_loader:
            if torch.cuda.is_available():
                imgs, masks = imgs.to(device), masks.to(device)
            outs = model(imgs)
            dices.append(dice_coefficient(outs, masks))
            ious.append(iou_score(outs, masks))

    print(f"Final Test Dice: {sum(dices)/len(dices):.4f}")
    print(f"Final Test IoU : {sum(ious)/len(ious):.4f}")

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["val_loss"],   label="Test loss")
    plt.xlabel("Epoch"); plt.ylabel("BCE-loss"); plt.title("Loss curves")
    plt.grid(True); plt.legend()

    plt.subplot(1, 2, 2)
    # Plot train and val dice
    if "train_dice" in history:
        plt.plot(epochs, history["train_dice"], label="Train Dice")
    plt.plot(epochs, history["val_dice"], label="Val Dice")
    plt.xlabel("Epoch"); plt.ylabel("Dice Score"); plt.title("Train/Val Dice")
    plt.grid(True); plt.legend()

    plt.tight_layout(); plt.show()

    # Visualize examples
    def make_rgba(mask: np.ndarray, color, alpha_val):
        """Return an RGBA image matching mask shape."""
        rgba          = np.zeros((*mask.shape, 4), dtype=float)
        rgba[..., :3] = color
        rgba[...,  3] = mask * alpha_val
        return rgba

    model.eval()
    with torch.no_grad():
        for b_idx, (images, masks) in enumerate(val_loader):
            if b_idx >= num_batches: break
            if torch.cuda.is_available():
                images = images.to(device)
            outs   = torch.sigmoid(model(images)).cpu()
            preds  = (outs > 0.5).float()

            for j in range(images.size(0)):
                img = images[j].cpu().permute(1, 2, 0).numpy()

                # Un-normalize the image to get pixel values in the original [0, 255] range
                # for brightness analysis. These are default ImageNet stats from Albumentations.
                unnormalized_img = (img * DATASET_STD) + DATASET_MEAN
                unnormalized_img = np.clip(unnormalized_img, 0, 1)

                gt = masks[j][0].cpu().numpy().astype(bool)
                pr = preds[j][0].numpy().astype(bool)

                # The image to display is the correctly denormalized one
                panel1 = unnormalized_img

                # For the overlays, use the same correctly denormalized image
                img_for_overlay = unnormalized_img

                # GT Overlay
                rgba_gt = make_rgba(gt, color=[0, 1, 0], alpha_val=alpha)

                # Prediction overlay
                TP = gt & pr
                FP = (~gt) & pr
                FN = gt & (~pr)

                comp_pred = np.zeros((*gt.shape, 4), dtype=float)
                comp_pred[TP] = [0, 1, 0, alpha]        # Green
                comp_pred[FP] = [1, 0, 0, alpha]        # Red
                comp_pred[FN] = [1, 0.5, 0, alpha]      # Orange

                canvas = np.zeros((*gt.shape, 4), dtype=float)  # Black background
                canvas[TP] = [0, 1, 0, 0.5]      # True positives - Green
                canvas[FP] = [1, 0, 0, 0.5]      # False positives - Red
                canvas[FN] = [1, 0.5, 0, 0.5]    # False negatives - Orange

                fig, axes = plt.subplots(1, 4, figsize=(18, 4))
                titles = ["Original",
                          "GT overlay (green)",
                          "Pred overlay\nTP=green  FP=red  FN=orange",
                          "Masks only\n(GT=green  Pred=orange)"]

                axes[0].imshow(panel1)

                axes[1].imshow(img_for_overlay)
                axes[1].imshow(rgba_gt)

                axes[2].imshow(img_for_overlay)
                axes[2].imshow(comp_pred)

                axes[3].imshow(canvas)

                for ax, t in zip(axes, titles):
                    ax.axis("off")
                    ax.set_title(t, fontsize=9)

                plt.tight_layout()

                # print(f"Image Analysis: {percent_zero:.2f}% of pixels are black (0), {percent_dark:.2f}% of pixels have intensity < 100.")
                
                plt.show()


def evaluate_classification(results: dict, num_batches: int = 1, class_names: list[str] | None = None) -> None:
    """Visualize training metrics and qualitative classification results.

    Args:
        results (dict): output from `train()` (s. train_generalized.py) mit
                        keys "model", "history", "val_loader", "device".
        num_batches (int, optional): wie viele *Validierungs*-Batches gezeigt werden.
        class_names (list[str], optional): Liste mit Klassennamen für Anzeige.
    """
    model      = results["model"]
    history    = results["history"]
    val_loader = results["val_loader"]
    device     = results["device"]


    epochs = range(len(history["train_loss"]))

    plt.figure(figsize=(12, 4))


    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["val_loss"],   label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss curves")
    plt.grid(True)
    plt.legend()


    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train acc")
    plt.plot(epochs, history["val_acc"],   label="Val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy curves")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


    model.eval()
    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            if batch_idx >= num_batches:
                break
            if torch.cuda.is_available():
                images = images.to(device)
            logits = model(images)
            probs  = softmax(logits).cpu()
            preds  = torch.argmax(probs, dim=1)

     
            batch_size = images.size(0)
            n = min(batch_size, 3)  
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            for i in range(3):
                ax = axes[i]
                ax.axis("off")
                if i < n:
                    img   = images[i].cpu().permute(1, 2, 0).numpy()
                    true  = labels[i].item()
                    pred  = preds[i].item()
                    prob  = probs[i, pred].item()

                    ax.imshow(img)
                    t = f"True: {true}"
                    p = f"Pred: {pred} ({prob:.2f})"
                    if class_names:
                        t = f"True: {class_names[true]}"
                        p = f"Pred: {class_names[pred]} ({prob:.2f})"
                    ax.set_title(f"{t}\n{p}", fontsize=9)

            plt.tight_layout()
            plt.show()

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            if torch.cuda.is_available(): 
                images = images.to(device)
                labels = labels.to(device).long()
            logits = model(images)
            preds  = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    final_acc = correct / total if total > 0 else 0.0
    print(f"Final Val Accuracy: {final_acc:.4f}")

def evaluate_losses(results_list, labels_list):
    """
    Evaluates and plots val dice scores for different loss functions

    Args:
        results_list: list of histories of a training runs
        labels_list: list of strings (name of loss used)
    """
    epochs = range(len(results_list[0]['history']['val_dice']))

    plt.figure(figsize=(10, 6))
    markers = ['o', 'x', '.', 's', '^', '*', '+']

    for i, results in enumerate(results_list):
        if i < len(markers):
            marker = markers[i]
        else:
            marker = 'o'

        plt.plot(epochs, results['history']['val_dice'], label=labels_list[i], marker=marker)

    plt.xlabel('Epochs')
    plt.ylabel('Validation Dice Score')
    plt.title('Comparison of Validation Dice Scores')
    plt.legend()
    plt.grid(True)
    plt.show()