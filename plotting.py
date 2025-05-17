import copy

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from util import shorten_label


def make_run_comparison_plot(epochs, accuracies_dict):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    fix, ax = plt.subplots(figsize=(5, 5))

    for label, accuracies in accuracies_dict.items():
        ax.plot(epochs, accuracies, label=f'{label} Accuracy')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
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
        display_label = shorten_label(label)
        arr = np.array(list(samples.values()))  # shape: (n_runs, n_epochs)
        trials, _ = arr.shape
        mean_vals = arr.mean(axis=0)
        lower_vals = np.percentile(arr, lower_pct, axis=0)
        upper_vals = np.percentile(arr, upper_pct, axis=0)
        mean_label = f'{display_label} Mean' if trials >= 2 else display_label
        update_steps = update_steps_dict[label]

        ax_train.plot(update_steps, mean_vals, label=mean_label)
        if trials >= 2:
            ax_train.fill_between(
                update_steps, lower_vals, upper_vals, alpha=0.3,
                label=f'{display_label} {int(100 * ci)}% CI'
            )

    ax_train.set_xlabel('Update step')
    ax_train.set_ylabel('Accuracy')
    min_final_train_acc = min([vs[-1] for _, samples in train_accuracies_dict.items() for vs in samples.values()])
    if min_final_train_acc > 0.9:
        ax_train.set_ylim(0.9, 1.0)  # Zoom in to 90 - 100 % area
    ax_train.set_title('Training Accuracy')
    ax_train.legend()
    ax_train.grid(True)
    ax_train.xaxis.set_major_locator(MaxNLocator(integer=True))

    for label, samples in val_accuracies_dict.items():
        display_label = shorten_label(label)
        arr = np.array(list(samples.values()))
        trials, _ = arr.shape
        mean_vals = arr.mean(axis=0)
        lower_vals = np.percentile(arr, lower_pct, axis=0)
        upper_vals = np.percentile(arr, upper_pct, axis=0)
        mean_label = f'{display_label} Mean' if trials >= 2 else display_label
        update_steps = update_steps_dict[label]

        ax_val.plot(update_steps, mean_vals, label=mean_label)
        if trials >= 2:
            ax_val.fill_between(
                update_steps, lower_vals, upper_vals, alpha=0.3,
                label=f'{display_label} {int(100 * ci)}% CI'
            )

    ax_val.set_xlabel('Update step')
    min_final_val_acc = min([vs[-1] for _, samples in val_accuracies_dict.items() for vs in samples.values()])
    if min_final_val_acc > 0.9:
        ax_val.set_ylim(0.9, 1.0)  # Zoom in to 90 - 100 % area
    ax_val.set_title('Validation Accuracy')
    ax_val.legend()
    ax_val.grid(True)
    ax_val.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    if plot_fname:
        fig.savefig(plot_fname)
    plt.show()


def plot_elapsed(baseline_label, times: dict[str, list[float]]):
    num_runs = len(times[baseline_label])

    labels = list(times.keys())

    times_no_baseline = copy.deepcopy(times)
    del times_no_baseline[baseline_label]
    diff_labels = list(times_no_baseline.keys())

    t_crit = stats.t.ppf(0.975, df=num_runs - 1)
    means = {}
    cis = {}
    for label, vals in times.items():
        arr = np.array(vals)
        means[label] = arr.mean()
        cis[label] = t_crit * arr.std(ddof=1) / np.sqrt(num_runs)

    base_arr = np.array(times[baseline_label])
    diffs = {}
    n_boot = 5000
    for label in diff_labels:
        arr = np.array(times[label])
        delta = arr - base_arr
        boot = np.random.choice(delta, size=(n_boot, num_runs), replace=True).mean(axis=1)
        low, high = np.percentile(boot, [2.5, 97.5])
        diffs[label] = (delta.mean(), delta.mean() - low, high - delta.mean())

    cmap = plt.get_cmap('tab10')
    colors = cmap.colors[:len(labels)]
    color_map = dict(zip(labels, colors))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for idx, label in enumerate(labels):
        arr = np.array(times[label])
        jitter = (np.random.rand(num_runs) - .5) * .2
        ax1.scatter(
            idx + jitter, arr,
            color=color_map[label], alpha=0.6
        )
        ax1.errorbar(
            idx, means[label],
            yerr=cis[label],
            fmt='o', color=color_map[label], capsize=5
        )

    ax1.plot(range(len(labels)), [means[l] for l in labels],
             linestyle='--', color='gray')
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('Elapsed time (s)')
    ax1.set_title('Timing across parameter settings')

    x = np.arange(1, len(labels))
    for xi, label in zip(x, diff_labels):
        m, err_low, err_high = diffs[label]
        ax2.errorbar(
            xi, m,
            yerr=[[err_low], [err_high]],
            fmt='o', color=color_map[label], capsize=5
        )

    ax2.axhline(0, linestyle='--', color='gray')
    ax2.set_xticks(x)
    ax2.set_xticklabels(diff_labels)
    ax2.set_ylabel(f'Mean difference vs {baseline_label} (s)')
    ax2.set_title('Bootstrap 95% CIs of differences')

    plt.tight_layout()
    plt.show()
