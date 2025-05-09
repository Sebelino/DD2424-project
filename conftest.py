import pytest

from determinism import Determinism # Must appear before any torch import
from datasets import DatasetParams
from training import TrainParams, NagParams


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
