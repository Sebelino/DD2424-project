from conftest import example_dataset_params, example_training_params  # Keep this import first
from datasets import make_datasets
from training import Trainer, FinishedAllEpochs, FinishedEpochs


def test_train_params(example_training_params):
    params_dict = example_training_params.minimal_dict()

    assert params_dict["n_epochs"] == 10
    assert "val_acc_target" not in params_dict  # Because it is None


def test_train_all_epochs(example_dataset_params, example_training_params):
    trainer = Trainer(example_training_params)
    labelled_train_loader, unlabelled_train_loader, val_loader = make_datasets(example_dataset_params, trainer.transform)
    trainer.load_dataset(labelled_train_loader, unlabelled_train_loader, val_loader)
    result = trainer.train(FinishedAllEpochs())

    assert result.validation_accuracies[-1] > 0.50


def test_training_reproducible(example_dataset_params, example_training_params, determinism):
    example_dataset_params.trainval_size = None
    example_training_params.n_epochs = 2  # Shorten time to run

    trainer1 = Trainer(example_training_params, determinism)
    train_loader, val_loader = make_datasets(example_dataset_params, trainer1.transform)
    trainer1.load_dataset(train_loader, val_loader)
    result1 = trainer1.train(FinishedAllEpochs())

    example_training_params.n_epochs = 3  # Number of epochs shouldn't affect the result
    trainer2 = Trainer(example_training_params, determinism)
    train_loader, val_loader = make_datasets(example_dataset_params, trainer2.transform)
    trainer2.load_dataset(train_loader, val_loader)
    result2 = trainer2.train(FinishedAllEpochs())

    expected = (0.8885869565217391, 0.9103260869565217)
    print(f"Expected: {expected}")
    print(f"Actual1:  {result1.validation_accuracies}")
    print(f"Actual2:  {result2.validation_accuracies[:2]}")

    # First test: Are the output of two consecutive trainings equal?
    assert result1.validation_accuracies == result2.validation_accuracies[:2]

    # Second test: Are the output across program executions equal?
    assert result1.validation_accuracies == expected


def test_train_each_epoch_individually(example_dataset_params, example_training_params):
    trainer = Trainer(example_training_params)
    labelled_train_loader, unlabelled_train_loader, val_loader = make_datasets(example_dataset_params, trainer.transform)
    trainer.load_dataset(labelled_train_loader, unlabelled_train_loader, val_loader)
    train_loader, val_loader = make_datasets(example_dataset_params, trainer.transform)

    result = trainer.train(FinishedAllEpochs())

    val_accs1 = result.validation_accuracies

    trainer = Trainer(example_training_params)
    labelled_train_loader, unlabelled_train_loader, val_loader = make_datasets(example_dataset_params, trainer.transform)
    trainer.load_dataset(labelled_train_loader, unlabelled_train_loader, val_loader)
    for epoch in range(example_training_params.n_epochs):
        result = trainer.train(FinishedEpochs(1))
    val_accs2 = result.validation_accuracies

    # Training for all epochs should be equivalent to training each epoch individually
    assert val_accs1 == val_accs2
