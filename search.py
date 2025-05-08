import logging

import optuna
from joblib import Memory
from optuna import Trial
from tqdm import tqdm

from datasets import DatasetParams, make_datasets
from training import TrainParams, Trainer, StopCondition

USE_CACHE = True

if USE_CACHE:
    memory = Memory("./runs/joblib_cache", verbose=0)
else:
    memory = Memory(location=None, verbose=0)


def objective(dataset_params: DatasetParams, params: TrainParams, trial: Trial):
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

    return objective_run(dataset_params, params, trial)


class FinishedOrPruned(StopCondition):
    def __init__(self, trial: Trial):
        self.trial = trial

    def remaining_steps(self, trainer: 'Trainer') -> int:
        max_update_steps = trainer.params.n_epochs * len(trainer.train_loader)
        print(f"max_update_steps={max_update_steps}")
        return max_update_steps

    def should_stop(self, trainer: 'Trainer') -> bool:
        if len(trainer.validation_accuracies) == 0:
            return False  # We haven't even run for a single epoch yet
        if trainer.epoch >= trainer.params.n_epochs:
            return True  # We have finished all epochs
        latest_val_acc = trainer.validation_accuracies[-1]
        stop_early = self.trial.should_prune()
        self.trial.report(latest_val_acc, trainer.epoch)
        if stop_early:
            tqdm.write(f"🔪 Trial {self.trial.number} pruned at epoch {trainer.epoch}")
        return stop_early


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            pass


@memory.cache(ignore=["trial"])
def objective_run(dataset_params, training_params, trial):
    trainer = Trainer(training_params, verbose=True)
    train_loader, val_loader = make_datasets(dataset_params, trainer.transform)
    trainer.load(train_loader, val_loader)
    result = trainer.train(stop_condition=FinishedOrPruned(trial))
    if len(result.epochs) < trainer.params.n_epochs:  # This means we stopped prematurely, before all epochs were finished
        raise optuna.TrialPruned()
    latest_val_acc = trainer.validation_accuracies[-1]
    return latest_val_acc


def run_search(dataset_params: DatasetParams, training_params: TrainParams, n_trials: int):
    tqdm.write("Running study with parameters:")
    tqdm.write(dataset_params.pprint())
    tqdm.write(training_params.pprint())
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
    optuna.logging.disable_default_handler()
    optuna.logging.set_verbosity(optuna.logging.INFO)

    # Grab the Optuna root logger
    optuna_logger = logging.getLogger("optuna")
    # Remove any existing handlers
    optuna_logger.handlers.clear()
    # Create and add our tqdm‐based handler
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    optuna_logger.addHandler(handler)

    study.optimize(lambda trial: objective(dataset_params, training_params, trial), n_trials=n_trials)

    return {
        "trial_accuracies": [t.value for t in study.trials],
        "best_trial": study.best_trial.number,
        "best_training_params": study.best_params,
        "best_validation_accuracy": study.best_value,
    }
