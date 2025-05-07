import pytest

from datasets import DatasetParams, make_datasets
from determinism import Determinism
from training import TrainParams, NagParams, Trainer, FinishedAllEpochs, FinishedEpochs


@pytest.fixture
def determinism():
    return Determinism(seed=42).sow()


@pytest.fixture
def example_dataset_params(determinism):
    return DatasetParams(
        splitting_seed=determinism.seed,
        shuffler_seed=determinism.seed,
        batch_size=32,
        trainval_size=100,  # Load a subset
        validation_set_fraction=0.2,  # 20 % of trainval set
    )


@pytest.fixture
def example_training_params(determinism) -> TrainParams:
    return TrainParams(
        seed=determinism.seed,
        architecture="resnet50",
        n_epochs=10,
        optimizer=NagParams(
            learning_rate=0.01,
            weight_decay=1e-4,
            momentum=0.9,
        ),
        freeze_layers=True,
        unfreezing_epochs=(3, 6),
        validation_freq=1,
        time_limit_seconds=None,
        val_acc_target=None,
    )


def test_train_params(example_training_params):
    params_dict = example_training_params.minimal_dict()

    assert params_dict["n_epochs"] == 10
    assert "val_acc_target" not in params_dict  # Because it is None


def test_train_all_epochs(example_dataset_params, example_training_params):
    trainer = Trainer(example_training_params)
    train_loader, val_loader = make_datasets(example_dataset_params, trainer.transform)
    trainer.load(train_loader, val_loader)
    result = trainer.train(FinishedAllEpochs())

    assert result.validation_accuracies[-1] > 0.50

def test_train_each_epoch_individually(example_dataset_params, example_training_params):
    trainer = Trainer(example_training_params)
    train_loader, val_loader = make_datasets(example_dataset_params, trainer.transform)
    trainer.load(train_loader, val_loader)
    result = trainer.train(FinishedAllEpochs())
    val_accs1 = result.validation_accuracies

    trainer = Trainer(example_training_params)
    train_loader, val_loader = make_datasets(example_dataset_params, trainer.transform)
    trainer.load(train_loader, val_loader)
    for epoch in range(example_training_params.n_epochs):
        result = trainer.train(FinishedEpochs(1))
    val_accs2 = result.validation_accuracies

    # Training for all epochs should be equivalent to training each epoch individually
    assert val_accs1 == val_accs2
