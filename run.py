import torch
from joblib import Memory
from torch.utils.data import DataLoader, random_split

from datasets import load_dataset
from determinism import Determinism
from training import TrainParams, Trainer, TrainingResult

use_cache = True
if use_cache:
    memory = Memory("./runs/joblib_cache", verbose=0)
else:
    memory = Memory(location=None, verbose=0)


def make_trainer(training_params: TrainParams):
    trainer = Trainer(training_params)

    train_dataset = load_dataset("trainval", trainer.transform)

    shrink_dataset = False
    if shrink_dataset:
        subset_size = 1000
        small_train_dataset = torch.utils.data.Subset(train_dataset, range(subset_size))
        train_dataset = small_train_dataset

    num_workers = 2

    # 80% train, 20% val split
    num_train = int(0.8 * len(train_dataset))
    num_val = len(train_dataset) - num_train
    torch_generator = torch.Generator()
    Determinism.torch_generator(training_params.seed, torch_generator)
    train_subset, val_subset = random_split(train_dataset, [num_train, num_val], generator=torch_generator)

    # DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=training_params.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=Determinism.data_loader_worker_init_fn(training_params.seed),
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=training_params.batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=Determinism.data_loader_worker_init_fn(training_params.seed),
    )

    return trainer, train_loader, val_loader


@memory.cache
def run(training_params: TrainParams) -> TrainingResult:
    trainer, train_loader, val_loader = make_trainer(training_params)
    result = trainer.train(train_loader, val_loader)
    return result
