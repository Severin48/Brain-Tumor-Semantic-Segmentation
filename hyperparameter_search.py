import optuna
import torch
from train_generalized_earlystopping import train, bce_loss, dice_loss, bce_dice_loss, focal_loss, tversky_loss
from data import load_mri_dataframe, get_dataloaders
from BaselineUNetParams import BaselineUNet
import torch.nn as nn

def objective(trial):

    # 1. Define hyperparameters to be optimized
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    num_layers = trial.suggest_int("num_layers", 2, 6) # Number of encoder/decoder layers
    num_filters = trial.suggest_categorical("num_filters", [8, 16, 32])
    lr_step_size = trial.suggest_int("step_size", 3, 7)  # for lr scheduler
    lr_gamma = trial.suggest_float("gamma", 0.1, 0.5)  # for lr scheduler

    # 2. Load data
    df = load_mri_dataframe()
    train_loader, val_loader = get_dataloaders(df, batch_size=batch_size, omit_empty_masks=True)

    # 3. Init model, optimizer, loss function
    model = BaselineUNet(num_layers=num_layers, num_filters=num_filters)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if optimizer_name == "Adam":
        optimizer_class = torch.optim.Adam
    else:
        optimizer_class = torch.optim.SGD

    #loss_fn_dict = {
    #    "bce_loss": bce_loss,
    #    "dice_loss": dice_loss,
    #    "bce_dice_loss": bce_dice_loss,
    #    "focal_loss": focal_loss,
    #    "tversky_loss": tversky_loss,
    #}
    #loss_fn = loss_fn_dict[loss_fn_name]

    # 4. Train model
    trained_model, results = train(
        model,
        train_loader,
        val_loader,
        device,
        lr=lr,
        optimizer_class=optimizer_class,
        loss_fn=bce_loss,
        epochs=300,
        lr_sched_cls=torch.optim.lr_scheduler.StepLR,
        lr_sched_kwargs={"step_size": lr_step_size, "gamma": lr_gamma},
    )

    # 5. Evaluate
    val_dice = results["history"]["val_dice"][-1]  # Last epoch dice score
    return val_dice

def create_study(n_trials=40):
    # Create a study object
    study = optuna.create_study(direction="maximize")  # We want to maximize the Dice score
    study.optimize(objective, n_trials=n_trials)  # Number of trials to run
    return study

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")  # Trial goal: maximize Dice score //TODO: combine with PatientPruner
    study.optimize(objective, n_trials=40)  # No. of trials to run                       //reason: optune will penalize trial if stopped early

    trial = study.best_trial

    print("\nBest Validation Dice Score: {}".format(trial.value))

    print("\nWith Parameters:")
    for key, value in trial.params.items():
        print("   {}: {}".format(key, value))