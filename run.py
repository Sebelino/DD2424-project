from typing import Dict

from joblib import Memory

from caching import invalidate_cache_entry
from datasets import make_datasets, DatasetParams
from determinism import Determinism
from training import TrainParams, Trainer, TrainingResult, FinishedAllEpochs

USE_CACHE = True

if USE_CACHE:
    memory = Memory("./runs/joblib_cache", verbose=0)
else:
    memory = Memory(location=None, verbose=0)


def run_multiple(
        dataset_params: DatasetParams,
        param_sets: Dict[str, TrainParams],
        determinism: Determinism,
        trials: int = 1,
        invalidate: bool = False
):
    dct = dict()
    for paramset_label, param_set in param_sets.items():
        param_set = param_set.copy()
        dct[paramset_label] = dict()
        for i in range(trials):
            param_set.seed += 1
            run_args = (dataset_params, param_set, determinism)
            invalidate_cache_entry(run, run_args, invalidate=invalidate)
            print(f"Running trial {i + 1}/{trials} for {paramset_label}")
            result = run(*run_args)
            run_label = f"Val acc seed={param_set.seed}"
            dct[paramset_label][run_label] = result
    return dct


def make_trained_trainer(dataset_params: DatasetParams, training_params: TrainParams, determinism: Determinism = None):
    trainer = Trainer(training_params, determinism)
    labelled_train_loader, unlabelled_train_loader, val_loader = make_datasets(
        dataset_params,
        trainer.base_transform,
        trainer.training_transform
    )
    trainer.load_dataset(labelled_train_loader, unlabelled_train_loader, val_loader)
    result = trainer.train(stop_condition=FinishedAllEpochs())
    return trainer, result


def try_loading_trainer(dataset_params: DatasetParams, training_params: TrainParams, determinism: Determinism = None):
    try:
        print("Trying to load trainer from disk...")
        trainer = Trainer.load(dataset_params, training_params, determinism)
    except FileNotFoundError:
        print("Trainer not found. Retraining...")
        trainer, _ = make_trained_trainer(dataset_params, training_params, determinism)
    return trainer


@memory.cache
def run(dataset_params: DatasetParams, training_params: TrainParams, determinism: Determinism = None) -> TrainingResult:
    trainer, result = make_trained_trainer(dataset_params, training_params, determinism)
    trainer.save(dataset_params)
    return result
