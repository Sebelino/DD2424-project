import os
import shutil
from typing import Dict

from tqdm.auto import tqdm

from datasets import DatasetParams
from plotting import make_run_comparison_plot, make_run_comparison_ci_plot
from run import run
from training import TrainingResult, TrainParams, Trainer


def evaluate_with_train_val_plot(result: TrainingResult):
    accuracies_dict = {
        "Training": result.training_accuracies,
        "Validation": result.validation_accuracies,
    }
    make_run_comparison_plot(result.epochs, accuracies_dict)


def evaluate_runs(results: Dict[str, TrainingResult]):
    accuracies_dict = {label: result.validation_accuracies for label, result in results.items()}
    epochs = list(results.values())[0].epochs
    for result in results.values():
        if result.epochs != epochs:
            raise ValueError(
                f"The runs are not comparable because the number of points differ: {len(epochs)} vs. {result.epochs}")
    make_run_comparison_plot(epochs, accuracies_dict)


def evaluate_runs_ci(results_per_paramset: Dict[str, Dict[str, TrainingResult]]):
    update_steps = dict()
    training_accuracies = dict()
    validation_accuracies = dict()
    for paramset_label, paramset_results_dict in results_per_paramset.items():
        update_steps_dict = {label: result.update_steps for label, result in paramset_results_dict.items()}
        update_steps_lst = list(update_steps_dict.values())[0]
        update_steps[paramset_label] = update_steps_lst
        train_acc_dict = {label: result.training_accuracies for label, result in paramset_results_dict.items()}
        training_accuracies[paramset_label] = train_acc_dict
        val_acc_dict = {label: result.validation_accuracies for label, result in paramset_results_dict.items()}
        validation_accuracies[paramset_label] = val_acc_dict
    make_run_comparison_ci_plot(update_steps, training_accuracies, validation_accuracies)


def run_with_different_seeds(dataset_params: DatasetParams, training_params: TrainParams, trials: int):
    training_params = training_params.copy()
    label_to_result = dict()
    for i in range(trials):
        training_params.seed += 1
        result = run(dataset_params, training_params)
        label = f"Val acc seed={training_params.seed}"
        label_to_result[label] = result
    evaluate_runs(label_to_result)


def run_comparison(dataset_params: DatasetParams, param_sets: Dict[str, TrainParams], trials: int = 1):
    dct = dict()
    for paramset_label, param_set in param_sets.items():
        param_set = param_set.copy()
        dct[paramset_label] = dict()
        for i in range(trials):
            param_set.seed += 1
            result = run(dataset_params, param_set)
            run_label = f"Val acc seed={param_set.seed}"
            dct[paramset_label][run_label] = result
    evaluate_runs_ci(dct)


def evaluate_test_accuracy_and_misclassified(trainer: Trainer, test_loader, test_dataset):
    from matplotlib import pyplot as plt
    import torch
    def collect_misclassified(model, loader, device, dataset):
        model.eval()
        correct = 0
        total = 0
        misclassified = []

        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(tqdm(loader, desc="Evaluating")):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                for i in range(inputs.size(0)):
                    if predicted[i] != labels[i]:
                        img_idx = idx * loader.batch_size + i
                        img_path = dataset._images[img_idx]
                        true_label = dataset._bin_labels[img_idx]
                        misclassified.append((img_path, true_label, predicted[i].item()))

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        model.train()
        return 100 * correct / total, misclassified

    # Directory for saving misclassified images
    misclassified_dir = "./misclassified"

    if os.path.exists(misclassified_dir):
        shutil.rmtree(misclassified_dir)
    os.makedirs(misclassified_dir, exist_ok=True)

    final_test_acc, misclassified_samples = collect_misclassified(trainer.model, test_loader,
                                                                  trainer.device,
                                                                  test_dataset)
    print(f"Final Test Accuracy: {final_test_acc:.2f}%")
    print(f"Number of misclassified samples: {len(misclassified_samples)}")

    def to_text(label: int):
        breed_names = [
            "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair", "Egyptian_Mau",
            "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue", "Siamese", "Sphynx",
            "american_bulldog", "american_pit_bull_terrier", "basset_hound", "beagle",
            "boxer", "chihuahua", "english_cocker_spaniel", "english_setter", "german_shorthaired",
            "great_pyrenees", "havanese", "japanese_chin", "keeshond", "leonberger",
            "miniature_pinscher", "newfoundland", "pomeranian", "pug", "saint_bernard",
            "samoyed", "scottish_terrier", "shiba_inu", "staffordshire_bull_terrier", "wheaten_terrier",
            "yorkshire_terrier"
        ]
        return breed_names[label]

    from PIL import Image

    # Plot the first 5 misclassified images
    num_to_plot = min(5, len(misclassified_samples))
    plt.figure(figsize=(15, 5))

    for i in range(num_to_plot):
        img_path, true_label, predicted_label = misclassified_samples[i]
        img = Image.open(img_path)
        plt.subplot(1, num_to_plot, i + 1)
        plt.imshow(img)
        plt.title(f"True: {to_text(true_label)}\nPred: {to_text(predicted_label)}")
        plt.axis('off')

    plt.show()

    for i, (img_path, true_label, predicted_label) in enumerate(misclassified_samples):
        filename = os.path.basename(img_path)
        new_filename = f"{i:04d}_true{true_label}_pred{predicted_label}_{filename}"
        shutil.copy(img_path, os.path.join(misclassified_dir, new_filename))

    print(f"Copied {len(misclassified_samples)} misclassified images to {misclassified_dir}")
