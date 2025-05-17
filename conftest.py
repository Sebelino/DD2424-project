import pytest

import augmentation
from augmentation import AugmentationParams
from datasets import DatasetParams
from determinism import Determinism  # Must appear before any torch import
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
        trainval_size=100,
        validation_set_fraction=0.2,
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
        augmentation=AugmentationParams(
            enabled=False,
            transform=augmentation.auto_transform("resnet50"),
            dropout_rate=None,
        ),
        validation_freq=1,
        freeze_layers=True,
        unfreeze_last_l_blocks=None,
        unfreezing_epochs=(3, 6),
        use_scheduler=False,
        scheduler_type="plateau",
        time_limit_seconds=None,
        val_acc_target=None,
    )
