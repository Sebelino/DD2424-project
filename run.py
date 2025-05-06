from joblib import Memory

from datasets import make_datasets, DatasetParams
from training import TrainParams, Trainer, TrainingResult

USE_CACHE = True

if USE_CACHE:
    memory = Memory("./runs/joblib_cache", verbose=0)
else:
    memory = Memory(location=None, verbose=0)


@memory.cache
def run(dataset_params: DatasetParams, training_params: TrainParams) -> TrainingResult:
    trainer = Trainer(training_params)
    train_loader, val_loader = make_datasets(dataset_params, trainer.transform)
    result = trainer.train(train_loader, val_loader)
    return result
