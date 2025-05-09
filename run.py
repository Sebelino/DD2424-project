from joblib import Memory

from datasets import make_datasets, DatasetParams
from determinism import Determinism
from training import TrainParams, Trainer, TrainingResult, FinishedAllEpochs

USE_CACHE = True

if USE_CACHE:
    memory = Memory("./runs/joblib_cache", verbose=0)
else:
    memory = Memory(location=None, verbose=0)


def make_trained_trainer(dataset_params: DatasetParams, training_params: TrainParams, determinism: Determinism = None):
    trainer = Trainer(training_params, determinism)
    train_loader, val_loader = make_datasets(dataset_params, trainer.transform)
    trainer.load_dataset(train_loader, val_loader)
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
