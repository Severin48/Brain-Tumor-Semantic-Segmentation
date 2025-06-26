import optuna
import torch
from train_generalized_earlystopping import train, bce_loss
from data import load_mri_dataframe, get_dataloaders_from_dfs, get_dataloaders_from_dfs_binary
from BaselineUNet import BaselineUNet
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from OriginalUNet import OriginalUNet
from sklearn.model_selection import train_test_split
from ImprovedUNet import ImprovedUNet
import json
import sys
import os

def objective(trial, augment=False, task='segmentation', model_type='baseline', df=None):
    # 1. Define hyperparameters to be optimized
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])


    lr_step_size = trial.suggest_int("step_size", 3, 7)
    lr_gamma = trial.suggest_float("gamma", 0.1, 0.5)

    if model_type == 'improved':    # dont optimize model parameters for original/baseline UNet
        num_layers = trial.suggest_int("num_layers", 3, 5)
        num_filters = trial.suggest_categorical("num_filters", [16, 32])

    if task == 'segmentation':
        df = df[df["diagnosis"] == 1].reset_index(drop=True)    # filter empty masks before doing the split

    df_trainval, df_test = train_test_split(df, test_size=0.15, random_state=48, stratify=df["diagnosis"])

    # 3. K-Fold cross-validation on trainval
    k = 3
    kf = KFold(n_splits=k, shuffle=True, random_state=48)
    val_dices = []
    val_dice_curves = []

    for train_idx, val_idx in kf.split(df_trainval):    # apply k-fold on training data only
        #-- Instantiate model on each fold --# 
        if model_type == 'original':
            model = OriginalUNet()
        else:
            if model_type == 'baseline':
                model = BaselineUNet()
            elif model_type == 'improved':
                model = ImprovedUNet(
                    num_layers=trial.params['num_layers'],
                    num_filters=trial.params['num_filters'],
                )

        df_train = df_trainval.iloc[train_idx].reset_index(drop=True)
        df_val = df_trainval.iloc[val_idx].reset_index(drop=True)

        # Select the correct dataloader based on the task
        if task == 'classification':
            train_loader, val_loader = get_dataloaders_from_dfs_binary(
                df_train,
                df_val,
                batch_size=batch_size,
                shuffle=True,
                augment=augment
            )
        else:  # segmentation
            train_loader, val_loader = get_dataloaders_from_dfs(
                df_train,
                df_val,
                batch_size=batch_size,
                shuffle=True,
                augment=augment
            )
        
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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
            epochs=30,
            task=task,
            lr_sched_cls=torch.optim.lr_scheduler.StepLR,
            lr_sched_kwargs={"step_size": lr_step_size, "gamma": lr_gamma},
        )
        if task == 'classification':
            # For classification, we use accuracy as the metric
            val_dices.append(results["history"]["val_acc"][-1])
            val_dice_curves.append(results["history"]["val_acc"])
        else:
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

def create_study(n_trials=30, df=None, task='segmentation', model_type='baseline', augment=False):
    study = optuna.create_study(direction="maximize")
    func = lambda trial: objective(trial, df=df, task=task, model_type=model_type, augment=augment)
    study.optimize(func, n_trials=n_trials)
    return study

if __name__ == "__main__":
    # Default params
    n_trials = 30
    model_type = 'improved'
    augment = True
    task = 'segmentation'
    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        try:
            if len(sys.argv) > 1:
                n_trials = int(sys.argv[1])
            if len(sys.argv) > 2:
                if sys.argv[2] not in ['segmentation', 'classification']:
                    raise ValueError("task must be 'segmentation' or 'classification'")
                task = sys.argv[2]
            if len(sys.argv) > 3:
                if sys.argv[3] not in ['improved', 'baseline', 'original']:
                    raise ValueError("model_type must be 'improved', 'baseline', or 'original'")
                model_type = sys.argv[3]
            if len(sys.argv) > 4:
                if sys.argv[4].lower() == 'true':
                    augment = True
                elif sys.argv[4].lower() == 'false':
                    augment = False
                else:
                    raise ValueError("augment must be True or False")
        except Exception as e:
            print("Usage: python hyperparameter_search.py [n_trials:int] [task:str] [model_type:str] [augment:True|False]")
            sys.exit(1)

    df = load_mri_dataframe()
    study = create_study(n_trials=n_trials, df=df, task=task, model_type=model_type, augment=augment)
    params = study.best_params
    trial = study.best_trial
    dice_val = trial.value
    print(f"Best trial: {trial.number} with value {dice_val}")
    best_params_dict = {
        "task": task,
        "model_type": model_type,
        "n_trials": n_trials,
        "augment": augment,
        "params": params,
        "dice_val/acc": dice_val
    }

    
    json_path = "best_params.json"
    with open(json_path, "r") as f:
        data = json.load(f)

    # Add results to best_params.json
    data.append(best_params_dict)

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)