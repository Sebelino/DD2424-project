import os
import shutil

from tqdm import tqdm

from plotting import make_train_val_plot
from training import TrainingResult


def evaluate_with_train_val_plot(result: TrainingResult):
    make_train_val_plot(result.epochs, result.training_accuracies, result.validation_accuracies)


def evaluate_test_accuracy_and_misclassified(result: TrainingResult, test_loader, test_dataset):
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

    trainer = result.trainer

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
