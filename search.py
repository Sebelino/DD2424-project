import optuna
from joblib import Memory
from tqdm import tqdm

from datasets import DatasetParams, make_datasets
from training import TrainParams, Trainer, FinishedEpochs

USE_CACHE = True

if USE_CACHE:
    memory = Memory("./runs/joblib_cache", verbose=0)
else:
    memory = Memory(location=None, verbose=0)


def objective(dataset_params: DatasetParams, params: TrainParams, trial):
    dataset_params = dataset_params.copy()
    params = params.copy()

    params.optimizer.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    params.optimizer.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    params.optimizer.momentum = trial.suggest_float("momentum", 0.7, 0.99, log=False)

    unfreeze_1 = trial.suggest_int("unfreeze_epoch_1", 1, 5)
    # Ensure second unfreeze is after first
    unfreeze_2 = trial.suggest_int(
        "unfreeze_epoch_2",
        low=unfreeze_1 + 1,
        high=9
    )
    params.unfreezing_epochs = (unfreeze_1, unfreeze_2)

    tqdm.write("Running trial with training parameters:")
    tqdm.write(params.pprint())
    return objective_run(dataset_params, params, trial)


@memory.cache(ignore=["trial"])
def objective_run(dataset_params, training_params, trial):
    trainer = Trainer(training_params, verbose=True)
    train_loader, val_loader = make_datasets(dataset_params, trainer.transform)
    trainer.load(train_loader, val_loader)
    result = None
    for epoch in range(1, training_params.n_epochs + 1):
        result = trainer.train(stop_condition=FinishedEpochs(1))
        val_acc = result.validation_accuracies[-1]
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return result.validation_accuracies[-1]


def run_search(dataset_params: DatasetParams, training_params: TrainParams, n_trials: int):
    # Create a study with TPE sampler and median pruner
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=training_params.seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=1,
            interval_steps=1
        )
    )

    study.optimize(lambda trial: objective(dataset_params, training_params, trial), n_trials=n_trials)

    return {
        "trial_accuracies": [t.value for t in study.trials],
        "best_trial": study.best_trial.number,
        "best_training_params": study.best_params,
        "best_validation_accuracy": study.best_value,
    }
