from joblib import Memory

from datasets import make_datasets, DatasetParams
from training import TrainParams, Trainer, TrainingResult, FinishedAllEpochs

USE_CACHE = True

if USE_CACHE:
    memory = Memory("./runs/joblib_cache", verbose=0)
else:
    memory = Memory(location=None, verbose=0)


@memory.cache
def run(dataset_params: DatasetParams, training_params: TrainParams) -> TrainingResult:
    trainer = Trainer(training_params)
    labelled_train_loader, unlabelled_train_loader, val_loader = make_datasets(dataset_params, trainer.transform)
    trainer.load(labelled_train_loader, unlabelled_train_loader, val_loader)
    result = trainer.train(stop_condition=FinishedAllEpochs())
    return result
