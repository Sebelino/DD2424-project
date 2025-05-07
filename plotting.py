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


def make_run_comparison_ci_plot(
        update_steps_dict,
        train_accuracies_dict,
        val_accuracies_dict,
        ci=0.95,
        plot_fname=None
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    alpha_pct = (100 - 100 * ci) / 2
    lower_pct = alpha_pct
    upper_pct = 100 - alpha_pct

    fig, (ax_train, ax_val) = plt.subplots(
        ncols=2, figsize=(12, 5), sharey=True
    )

    for label, samples in train_accuracies_dict.items():
        arr = np.array(list(samples.values()))  # shape: (n_runs, n_epochs)
        trials, _ = arr.shape
        mean_vals = arr.mean(axis=0)
        lower_vals = np.percentile(arr, lower_pct, axis=0)
        upper_vals = np.percentile(arr, upper_pct, axis=0)
        mean_label = f'{label} Mean' if trials >= 2 else label
        update_steps = update_steps_dict[label]

        ax_train.plot(update_steps, mean_vals, label=mean_label)
        if trials >= 2:
            ax_train.fill_between(
                update_steps, lower_vals, upper_vals, alpha=0.3,
                label=f'{label} {int(100 * ci)}% CI'
            )

    ax_train.set_xlabel('Update step')
    ax_train.set_ylabel('Accuracy (%)')
    ax_train.set_title('Training Accuracy')
    ax_train.legend()
    ax_train.grid(True)
    ax_train.xaxis.set_major_locator(MaxNLocator(integer=True))

    for label, samples in val_accuracies_dict.items():
        arr = np.array(list(samples.values()))
        trials, _ = arr.shape
        mean_vals = arr.mean(axis=0)
        lower_vals = np.percentile(arr, lower_pct, axis=0)
        upper_vals = np.percentile(arr, upper_pct, axis=0)
        mean_label = f'{label} Mean' if trials >= 2 else label
        update_steps = update_steps_dict[label]

        ax_val.plot(update_steps, mean_vals, label=mean_label)
        if trials >= 2:
            ax_val.fill_between(
                update_steps, lower_vals, upper_vals, alpha=0.3,
                label=f'{label} {int(100 * ci)}% CI'
            )

    ax_val.set_xlabel('Update step')
    ax_val.set_title('Validation Accuracy')
    ax_val.legend()
    ax_val.grid(True)
    ax_val.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    if plot_fname:
        fig.savefig(plot_fname)
    plt.show()
