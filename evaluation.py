import os
import shutil
from dataclasses import asdict
from typing import Dict, Any

import numpy as np
from joblib import Memory
from scipy.stats import stats
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from augmentation import AugmentationParams
from caching import invalidate_cache_entry
from datasets import DatasetParams, load_dataset
from determinism import Determinism
from freezing import MaskedFineTuningParams
from plotting import make_run_comparison_plot, make_run_comparison_ci_plot, plot_elapsed
from run import run, run_multiple, try_loading_trainer
from training import TrainingResult, TrainParams, Trainer, NagParams
from util import shorten_label, suppress_weights_only_warning

USE_CACHE = True

if USE_CACHE:
    memory = Memory("./runs/joblib_cache", verbose=0)
else:
    memory = Memory(location=None, verbose=0)


def make_paramset_string(params: dict) -> str:
    parts = []

    def _flatten(d, prefix=""):
        for key in sorted(d):
            val = d[key]
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(val, dict):
                _flatten(val, full_key)
            else:
                parts.append(f"{full_key}={val}")

    _flatten(params)
    return ",".join(parts)


def tweak(params: TrainParams, overrides: dict[str, Any]):
    params = params.copy()
    for k, v in overrides.items():
        if isinstance(v, dict):
            if k == "augmentation":
                v = AugmentationParams(**{**asdict(params.augmentation), **v})
            elif k == "mft":
                v = MaskedFineTuningParams(**{**asdict(params.mft), **v})
            elif k == "optimizer" and params.optimizer.name == "nag":
                v = NagParams(**{**asdict(params.optimizer), **v})
            else:
                raise NotImplementedError
        setattr(params, k, v)
    return params


def override_param_sets(training_params: TrainParams, overrides: list[dict[str, Any]] | dict[str, dict[str, Any]]):
    if isinstance(overrides, list):
        param_sets = {make_paramset_string(o): tweak(training_params, o) for o in overrides}
    elif isinstance(overrides, dict):
        param_sets = {label: tweak(training_params, param_set) for label, param_set in overrides.items()}
    else:
        raise ValueError()
    return param_sets


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


def evaluate_runs_print(results_per_paramset: Dict[str, Dict[str, TrainingResult]]):
    for label, paramset_result in results_per_paramset.items():
        training_elapseds = {l: r.training_elapsed for l, r in paramset_result.items()}
        arr = np.array(list(training_elapseds.values()))
        mean_training_elapsed = arr.mean(axis=0)
        print(f"Elapsed training time: {mean_training_elapsed} for {shorten_label(label)}")
    for label, paramset_result in results_per_paramset.items():
        val_accs = {l: r.validation_accuracies for l, r in paramset_result.items()}
        arr = np.array(list(val_accs.values()))
        mean_val_accs = arr.mean(axis=0)
        print(f"Final mean val acc: {100 * mean_val_accs[-1]:.2f} % for {shorten_label(label)}")
    for label, paramset_result in results_per_paramset.items():
        val_accs = {l: r.validation_accuracies for l, r in paramset_result.items()}
        arr = np.array(list(val_accs.values()))
        mean_val_accs = arr.mean(axis=0)
        print(f"Max   mean val acc: {100 * max(mean_val_accs):.2f} % for {shorten_label(label)}")


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


def run_with_different_seeds(
        dataset_params: DatasetParams,
        training_params: TrainParams,
        determinism: Determinism,
        trials: int
):
    training_params = training_params.copy()
    label_to_result = dict()
    for _ in range(trials):
        result = run(dataset_params, training_params, determinism)
        label = f"Val acc seed={training_params.seed}"
        label_to_result[label] = result
        training_params.seed += 1
    evaluate_runs(label_to_result)


def run_comparison(dataset_params: DatasetParams, param_sets: Dict[str, TrainParams], trials: int = 1):
    results = run_multiple(dataset_params, param_sets, trials)
    evaluate_runs_ci(results)


def run_dataset_comparison(param_sets: Dict[str, DatasetParams], training_params: TrainParams, trials: int = 1):
    dct = dict()
    for paramset_label, param_set in param_sets.items():
        param_set = param_set.copy()
        dct[paramset_label] = dict()
        for i in range(trials):
            result = run(param_set, training_params)
            run_label = f"Val acc seed={training_params.seed}"
            dct[paramset_label][run_label] = result
    evaluate_runs_ci(dct)


def evaluate_predictions(trainer: Trainer, test_loader, test_dataset):
    import torch
    predicted_labels = []
    true_labels = []
    image_paths = []

    trainer.model.eval()

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
            inputs, labels = inputs.to(trainer.device), labels.to(trainer.device)
            outputs = trainer.model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            for i in range(inputs.size(0)):
                img_idx = idx * test_loader.batch_size + i
                predicted_labels.append(predicted[i].item())
                true_labels.append(test_dataset._labels[img_idx])
                image_paths.append(test_dataset._images[img_idx])

    trainer.model.train()
    return predicted_labels, true_labels, image_paths


def evaluate_test_accuracy(trainer: Trainer, test_loader):
    import torch
    model = trainer.model
    loader = test_loader
    device = trainer.device

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    model.train()
    return 100.0 * correct / total


def evaluate_test_accuracy_and_misclassified(trainer: Trainer, test_loader, test_dataset):
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

    final_test_acc, misclassified_samples = collect_misclassified(
        trainer.model,
        test_loader,
        trainer.device,
        test_dataset
    )
    return final_test_acc, misclassified_samples


def show_misclassified(misclassified_samples):
    from matplotlib import pyplot as plt
    from PIL import Image
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

    # Directory for saving misclassified images
    misclassified_dir = "./misclassified"

    if os.path.exists(misclassified_dir):
        shutil.rmtree(misclassified_dir)
    os.makedirs(misclassified_dir, exist_ok=True)

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


def evaluate_elapsed_time(results_per_paramset: Dict[str, Dict[str, TrainingResult]], baseline_label: str):
    times = {psl: [r.training_elapsed for _, r in v.items()] for psl, v in results_per_paramset.items()}
    plot_elapsed(baseline_label, times)


def evaluate_final_test_accuracy(
        dataset_params: DatasetParams,
        training_params: TrainParams,
        determinism: Determinism,
        trials: int = 1,
        display_misclassified=False,
        invalidate=False,
):
    suppress_weights_only_warning()
    dataset_params = dataset_params.copy()
    # train on the full train+val set
    dataset_params.validation_set_fraction = 0
    training_params = training_params.copy()
    # don't compute val acc while training
    training_params.validation_freq = 0
    test_dataset = load_dataset("test", Trainer.make_base_transform(training_params), dataset_params.target_types)
    print(f"Test size: {len(test_dataset)}")

    misclassified_samples = []
    test_accs = []
    for i in range(trials):
        args = (dataset_params, training_params, determinism, display_misclassified, test_dataset)
        invalidate_cache_entry(evaluate_cached, args, invalidate=invalidate)
        test_acc, misclassified_samples = evaluate_cached(dataset_params, training_params, determinism,
                                                          display_misclassified, test_dataset)
        test_accs.append(test_acc)
        training_params.seed += 1

    test_acc_mean = np.mean(test_accs)
    print(f"Test Accuracy Mean: {test_acc_mean:.2f} %")
    if len(test_accs) >= 2:
        test_acc_se = stats.sem(np.array(test_accs))
        print(f"Test Accuracy Standard Error: {test_acc_se:.2f} percentage points")

    if display_misclassified:
        print(f"Number of misclassified samples: {len(misclassified_samples)}")
        show_misclassified(misclassified_samples)


@memory.cache(ignore=["test_dataset"])
def evaluate_cached(
        dataset_params: DatasetParams,
        training_params: TrainParams,
        determinism: Determinism,
        display_misclassified: bool,
        test_dataset
):
    test_loader = DataLoader(
        test_dataset,
        batch_size=dataset_params.batch_size,  # default 1
        # shuffle=False, #default False
        # num_workers=0, #default 0
        # persistent_workers=False, #default False
        pin_memory=True,
        # worker_init_fn does not get called if num_workers=0
        # worker_init_fn=Determinism.data_loader_worker_init_fn(dataset_params.shuffler_seed),
    )

    misclassified_samples = []
    trainer = try_loading_trainer(dataset_params, training_params, determinism)
    if display_misclassified:
        test_acc, misclassified_samples = evaluate_test_accuracy_and_misclassified(
            trainer,
            test_loader,
            test_dataset
        )
    else:
        test_acc = evaluate_test_accuracy(trainer, test_loader)  # deterministic
    print(f"Test Accuracy: {test_acc:.3f} %")
    return test_acc, misclassified_samples
