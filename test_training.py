import pytest

from determinism import Determinism
from training import TrainParams, NagParams


@pytest.fixture
def example_training_params() -> TrainParams:
    determinism = Determinism(seed=42)
    return TrainParams(
        seed=determinism.seed,
        batch_size=32,
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

    assert params_dict["batch_size"] == 32
    assert "val_acc_target" not in params_dict # Because it is None
