import numpy as np


def make_train_val_plot(epochs, train_accuracies, val_accuracies):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    fix, ax = plt.subplots(figsize=(5, 5))

    ax.plot(epochs, train_accuracies, label='Train Accuracy')
    ax.plot(epochs, val_accuracies, label='Validation Accuracy')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Train vs Validation Accuracy')
    ax.legend()
    ax.grid(True)

    ax.set_box_aspect(1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.show()


def make_run_comparison_plot(epochs, accuracies_dict):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    fix, ax = plt.subplots(figsize=(5, 5))

    for label, accuracies in accuracies_dict.items():
        ax.plot(epochs, accuracies, label=f'{label} Accuracy')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy comparison')
    ax.legend()
    ax.grid(True)

    ax.set_box_aspect(1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.show()


def make_run_comparison_ci_plot(epochs, accuracies_samples_dict, ci=95):
    from matplotlib.ticker import MaxNLocator
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    alpha_pct = (100 - ci) / 2
    lower_pct = alpha_pct
    upper_pct = 100 - alpha_pct

    for label, samples in accuracies_samples_dict.items():
        arr = np.array(samples)
        mean_vals = arr.mean(axis=0)
        lower_vals = np.percentile(arr, lower_pct, axis=0)
        upper_vals = np.percentile(arr, upper_pct, axis=0)

        ax.plot(epochs, mean_vals, label=f'{label} Mean')
        ax.fill_between(epochs, lower_vals, upper_vals, alpha=0.3,
                        label=f'{label} {ci}% CI')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy Comparison with Confidence Intervals')
    ax.legend()
    ax.grid(True)

    ax.set_box_aspect(1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.show()
