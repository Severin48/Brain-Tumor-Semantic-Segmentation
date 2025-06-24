import optuna
import torch
from train_generalized_earlystopping import train, bce_loss
from data import load_mri_dataframe, get_dataloaders, get_dataloaders_from_dfs
from BaselineUNetParams import BaselineUNet
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from OriginalUNet import OriginalUNet
from sklearn.model_selection import train_test_split

def objective(trial, augment=False, task='segmentation', original_model=False, df=None):
    # 1. Define hyperparameters to be optimized
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])

    if original_model:   # dont optimize model parameters for original UNet
        model = OriginalUNet()
    else:       
        num_layers = trial.suggest_int("num_layers", 2, 5)
        num_filters = trial.suggest_categorical("num_filters", [8, 16, 32])
        model = BaselineUNet(num_layers=num_layers, num_filters=num_filters)

    lr_step_size = trial.suggest_int("step_size", 3, 7)
    lr_gamma = trial.suggest_float("gamma", 0.1, 0.5)

    if task == 'segmentation':
        df = df[df["diagnosis"] == 1].reset_index(drop=True)    # filter empty masks before doing the split

    df_trainval, df_test = train_test_split(df, test_size=0.15, random_state=48, stratify=df["diagnosis"])

    # 3. K-Fold cross-validation on trainval
    k = 3
    kf = KFold(n_splits=k, shuffle=True, random_state=48)
    val_dices = []
    val_dice_curves = []

    for train_idx, val_idx in kf.split(df_trainval):    # apply k-fold on training data only
        df_train = df_trainval.iloc[train_idx].reset_index(drop=True)
        df_val = df_trainval.iloc[val_idx].reset_index(drop=True)

        train_loader, val_loader = get_dataloaders_from_dfs(    # get dataloaders, but no splits
            df_train,
            df_val,
            batch_size=batch_size,
            shuffle=True,
            augment=augment
        )
        device = torch.device("cuda")
        model.to(device)
        optimizer_class = torch.optim.Adam if optimizer_name == "Adam" else torch.optim.SGD

        _, results = train(
            model,
            train_loader,
            val_loader, # this is the actual validation set (not test set!)
            device,
            lr=lr,
            optimizer_class=optimizer_class,
            loss_fn=bce_loss,
            epochs=100,
            task=task,
            lr_sched_cls=torch.optim.lr_scheduler.StepLR,
            lr_sched_kwargs={"step_size": lr_step_size, "gamma": lr_gamma},
        )
        val_dices.append(results["history"]["val_dice"][-1])
        val_dice_curves.append(results["history"]["val_dice"])

    # Pad curves to the same length (in case of early stopping)
    max_len = max(len(curve) for curve in val_dice_curves)
    for i in range(len(val_dice_curves)):
        if len(val_dice_curves[i]) < max_len:
            val_dice_curves[i] += [val_dice_curves[i][-1]] * (max_len - len(val_dice_curves[i]))

    # Average per-epoch validation dice across folds
    cv_val_dice = np.mean(val_dice_curves, axis=0)

    # 4. Return mean validation Dice across folds, and store cv_val_dice in user_attrs for later access
    trial.set_user_attr("cv_val_dice", cv_val_dice.tolist())
    return np.mean(val_dices)

def create_study(n_trials=30, df=None, task='segmentation', original_model=False, augment=False):
    study = optuna.create_study(direction="maximize")
    func = lambda trial: objective(trial, df=df, task='segmentation', original_model=original_model, augment=augment)
    study.optimize(func, n_trials=n_trials)
    return study

if __name__ == "__main__":
    import pandas as pd
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    trial = study.best_trial
    print("\nBest Validation Dice Score (CV mean): {:.4f}".format(trial.value))
    print("\nWith Parameters:")
    for key, value in trial.params.items():
        print("   {}: {}".format(key, value))

    # --- Final evaluation on held-out test set ---
    print("\nEvaluating best hyperparameters on held-out test set...")
    # Reload data and split as before
    df = load_mri_dataframe()
    from sklearn.model_selection import train_test_split
    df_trainval, df_test = train_test_split(df, test_size=0.15, random_state=48, stratify=df["diagnosis"])

    # Use all trainval for training, test for evaluation
    train_loader, _ = get_dataloaders(df_trainval, batch_size=trial.params["batch_size"], val_split=0.0, omit_empty_masks=True)
    _, test_loader = get_dataloaders(df_test, batch_size=trial.params["batch_size"], val_split=0.0, omit_empty_masks=True)

    model = BaselineUNet(
        num_layers=trial.params["num_layers"],
        num_filters=trial.params["num_filters"]
    )
    device = torch.device("cuda")
    model.to(device)
    optimizer_class = torch.optim.Adam if trial.params["optimizer"] == "Adam" else torch.optim.SGD

    _, results = train(
        model,
        train_loader,
        test_loader,
        device,
        lr=trial.params["lr"],
        optimizer_class=optimizer_class,
        loss_fn=bce_loss,
        epochs=100,
        lr_sched_cls=torch.optim.lr_scheduler.StepLR,
        lr_sched_kwargs={"step_size": trial.params["step_size"], "gamma": trial.params["gamma"]},
    )
    test_dice = results["history"]["val_dice"][-1]
    print(f"\nTest Dice Score: {test_dice:.4f}")