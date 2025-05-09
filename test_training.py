from datasets import make_datasets
from conftest import example_dataset_params, example_training_params
from training import Trainer, FinishedAllEpochs, FinishedEpochs


def test_train_params(example_training_params):
    params_dict = example_training_params.minimal_dict()

    assert params_dict["n_epochs"] == 10
    assert "val_acc_target" not in params_dict  # Because it is None


def test_train_all_epochs(example_dataset_params, example_training_params):
    trainer = Trainer(example_training_params)
    labelled_train_loader, unlabelled_train_loader, val_loader = make_datasets(example_dataset_params, trainer.transform)
    trainer.load(labelled_train_loader, unlabelled_train_loader, val_loader)
    result = trainer.train(FinishedAllEpochs())

    assert result.validation_accuracies[-1] > 0.50


def test_train_each_epoch_individually(example_dataset_params, example_training_params):
    trainer = Trainer(example_training_params)
    labelled_train_loader, unlabelled_train_loader, val_loader = make_datasets(example_dataset_params, trainer.transform)
    trainer.load(labelled_train_loader, unlabelled_train_loader, val_loader)

    result = trainer.train(FinishedAllEpochs())

    val_accs1 = result.validation_accuracies

    trainer = Trainer(example_training_params)
    labelled_train_loader, unlabelled_train_loader, val_loader = make_datasets(example_dataset_params, trainer.transform)
    trainer.load(labelled_train_loader, unlabelled_train_loader, val_loader)
    for epoch in range(example_training_params.n_epochs):
        result = trainer.train(FinishedEpochs(1))
    val_accs2 = result.validation_accuracies

    # Training for all epochs should be equivalent to training each epoch individually
    assert val_accs1 == val_accs2
