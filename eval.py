import torch
import matplotlib.pyplot as plt
from util import *


def evaluate(results, num_batches=1):  # Using results returned by train()
    """
    Plot training metrics and visualize predictions on validation set.

    Args:
        results: dict from train(), must contain model, history, val_loader, device
        num_batches: how many validation batches to display
    """
    model = results['model']
    history = results['history']
    val_loader = results['val_loader']
    device = results['device']

    # Plot loss and metrics
    plt.figure()
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_dice'], label='Val Dice')
    plt.plot(history['val_iou'], label='Val IoU')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Training Metrics')
    plt.grid(True)
    plt.show()

    # Visualize predictions
    model.eval()
    with torch.no_grad():
        for i, (images, masks) in enumerate(val_loader):
            if i >= num_batches:
                break
            images = images.to(device)
            outputs = torch.sigmoid(model(images)).cpu()
            preds = (outputs > 0.5).float()

            for j in range(images.size(0)):
                img = images[j].cpu().permute(1,2,0).numpy()
                mask = masks[j][0].cpu().numpy()
                pred = preds[j][0].numpy()

                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(img)
                axes[0].set_title('Image')
                axes[0].axis('off')

                axes[1].imshow(mask, cmap='gray')
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')

                axes[2].imshow(pred, cmap='gray')
                axes[2].set_title('Prediction')
                axes[2].axis('off')
                plt.tight_layout()
                plt.show()

    # Compute final numbers on full validation set
    dices, ious = [], []
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            dices.append(dice_coefficient(outputs, masks))
            ious.append(iou_score(outputs, masks))
    print(f"Final Val Dice: {sum(dices)/len(dices):.4f}")
    print(f"Final Val IoU: {sum(ious)/len(ious):.4f}")
