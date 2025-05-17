from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from util import dice_coefficient, iou_score


def evaluate(results: dict, num_batches: int = 1) -> None:
    """Visualize training metrics and qualitative results.

    Args:
        results (dict): output from `train()` in train.py.
        num_batches (int, optional): how many *validation* batches to display.
    """

    model      = results["model"]
    history    = results["history"]
    val_loader = results["val_loader"]
    device     = results["device"]

    # Plots
    epochs = range(len(history["train_loss"]))

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["val_loss"],   label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE-loss")
    plt.title("Loss curves")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_dice"], label="Val Dice")
    plt.plot(epochs, history["val_iou"],  label="Val IoU")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation metrics")
    plt.grid(True)
    plt.legend()

    plt.tight_layout(); plt.show()

    # Predictions
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            if batch_idx >= num_batches: break
            images = images.to(device)
            outs   = torch.sigmoid(model(images)).cpu()
            preds  = (outs > 0.5).float()

            for j in range(images.size(0)):
                img  = images[j].cpu().permute(1, 2, 0).numpy()
                mask = masks[j][0].cpu().numpy()
                pred = preds[j][0].numpy()

                fig, ax = plt.subplots(1, 3, figsize=(12, 4))
                ax[0].imshow(img);  ax[0].set_title("Image");         ax[0].axis("off")
                ax[1].imshow(mask, cmap="gray"); ax[1].set_title("Ground truth"); ax[1].axis("off")
                ax[2].imshow(pred, cmap="gray"); ax[2].set_title("Prediction");   ax[2].axis("off")
                plt.tight_layout(); plt.show()

    # Scores on val dataset
    dices, ious = [], []
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outs        = model(imgs)
            dices.append(dice_coefficient(outs, masks))
            ious.append(iou_score(outs, masks))
    print(f"Final Val Dice: {sum(dices)/len(dices):.4f}")
    print(f"Final Val IoU : {sum(ious)/len(ious):.4f}")

