"""
Visualization script for data size experiment results

Creates plots to show how accuracy changes with different dataset sizes.
This is for the final report/presentation.
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_accuracy_vs_data_size(results, output_dir):
    """Main plot showing accuracy vs dataset size"""

    # extract data from results
    fractions = [r['fraction'] * 100 for r in results]  # convert to percentages
    train_sizes = [r['train_size'] for r in results]
    test_accs = [r['test_acc'] * 100 for r in results]  # convert to percentages

    # also get validation accuracy if it exists
    val_accs = []
    for r in results:
        if r['best_val_acc'] is not None:
            val_accs.append(r['best_val_acc'] * 100)
        else:
            val_accs.append(0)

    # create two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: accuracy vs percentage of data used
    ax1.plot(fractions, val_accs, 'o-', label='Validation Accuracy', linewidth=2, markersize=8)
    ax1.plot(fractions, test_accs, 's-', label='Test Accuracy', linewidth=2, markersize=8)
    ax1.set_xlabel('Percentage of Training Data Used (%)', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Model Performance vs. Dataset Size', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.set_ylim([0, 105])

    # add labels to show actual accuracy values
    for i in range(len(fractions)):
        ax1.annotate(f'{test_accs[i]:.1f}%',
                    (fractions[i], test_accs[i]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center',
                    fontsize=9)

    # Right plot: accuracy vs absolute number of training samples
    ax2.plot(train_sizes, val_accs, 'o-', label='Validation Accuracy', linewidth=2, markersize=8)
    ax2.plot(train_sizes, test_accs, 's-', label='Test Accuracy', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Training Samples', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Model Performance vs. Absolute Dataset Size', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.set_ylim([0, 105])

    plt.tight_layout()

    # save as high resolution image
    output_path = output_dir / 'accuracy_vs_data_size.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def plot_learning_curves(results, output_dir):
    """Plot training loss and validation accuracy over epochs for each data fraction"""

    # create a 2x2 grid for up to 4 experiments
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, result in enumerate(results):
        if idx >= 4:  # only plot first 4 experiments
            break

        history = result['history']
        epochs = [h['epoch'] for h in history]
        train_losses = [h['train_loss'] for h in history]

        # get validation accuracies (convert to percentage)
        val_accs = []
        for h in history:
            if h['val_acc'] is not None:
                val_accs.append(h['val_acc'] * 100)
            else:
                val_accs.append(0)

        ax = axes[idx]

        # plot training loss on left y-axis
        color = 'tab:red'
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Training Loss', color=color, fontsize=11)
        ax.plot(epochs, train_losses, 'o-', color=color, linewidth=2, markersize=6)
        ax.tick_params(axis='y', labelcolor=color)
        ax.grid(True, alpha=0.3)

        # plot validation accuracy on right y-axis
        ax2 = ax.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Validation Accuracy (%)', color=color, fontsize=11)
        ax2.plot(epochs, val_accs, 's-', color=color, linewidth=2, markersize=6)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim([0, 105])

        # title showing which data fraction this is
        fraction_pct = int(result['fraction'] * 100)
        train_size = result['train_size']
        ax.set_title(f'{fraction_pct}% of Data ({train_size} samples)',
                    fontsize=12, fontweight='bold')

    plt.tight_layout()
    output_path = output_dir / 'learning_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def plot_data_efficiency(results, output_dir):
    """
    Show diminishing returns - how much accuracy we gain per additional data.
    This helps answer "is it worth collecting more data?"
    """

    train_sizes = [r['train_size'] for r in results]
    test_accs = [r['test_acc'] * 100 for r in results]

    # calculate marginal gain: how much accuracy we gain per additional sample
    marginal_gains = []
    for i in range(1, len(train_sizes)):
        size_diff = train_sizes[i] - train_sizes[i-1]
        acc_diff = test_accs[i] - test_accs[i-1]

        if size_diff > 0:
            # scale to per 1000 samples to make numbers readable
            gain_per_1000 = (acc_diff / size_diff) * 1000
            marginal_gains.append(gain_per_1000)
        else:
            marginal_gains.append(0)

    # create labels for the x-axis showing the transition range
    x_labels = []
    for i in range(len(marginal_gains)):
        from_pct = int(results[i]['fraction'] * 100)
        to_pct = int(results[i+1]['fraction'] * 100)
        x_labels.append(f'{from_pct}% → {to_pct}%')

    fig, ax = plt.subplots(figsize=(10, 6))

    # bar chart showing marginal gains
    bars = ax.bar(x_labels, marginal_gains, color='steelblue', alpha=0.7, edgecolor='black')

    # add value labels on top of bars
    for bar, gain in zip(bars, marginal_gains):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{gain:.3f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Data Increase Range', fontsize=12)
    ax.set_ylabel('Accuracy Gain per 1000 Samples (%)', fontsize=12)
    ax.set_title('Data Efficiency: Marginal Returns of Additional Training Data',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / 'data_efficiency.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def generate_summary_table(results, output_dir):
    """Create a text file with summary statistics - useful for the report"""

    output_path = output_dir / 'summary_table.txt'

    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DATA SIZE EXPERIMENT SUMMARY\n")
        f.write("="*80 + "\n\n")

        # table header
        f.write(f"{'Data %':<10} {'Train Size':<12} {'Val Size':<10} "
                f"{'Val Acc %':<12} {'Test Acc %':<12} {'Epochs':<8}\n")
        f.write("-"*80 + "\n")

        # table rows
        for r in results:
            fraction_pct = int(r['fraction'] * 100)
            val_acc = r['best_val_acc'] * 100 if r['best_val_acc'] else 0
            test_acc = r['test_acc'] * 100

            f.write(f"{fraction_pct:<10} {r['train_size']:<12} {r['val_size']:<10} "
                   f"{val_acc:<12.2f} {test_acc:<12.2f} {r['epochs']:<8}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("KEY FINDINGS:\n")
        f.write("="*80 + "\n\n")

        # find best and worst results
        best_result = max(results, key=lambda x: x['test_acc'])
        worst_result = min(results, key=lambda x: x['test_acc'])

        f.write(f"1. Best Test Accuracy: {best_result['test_acc']*100:.2f}% ")
        f.write(f"(using {int(best_result['fraction']*100)}% of data)\n\n")

        f.write(f"2. Worst Test Accuracy: {worst_result['test_acc']*100:.2f}% ")
        f.write(f"(using {int(worst_result['fraction']*100)}% of data)\n\n")

        f.write(f"3. Accuracy Range: {(best_result['test_acc'] - worst_result['test_acc'])*100:.2f}%\n\n")

        # compare full data vs minimal data
        full_data = None
        min_data = None
        for r in results:
            if r['fraction'] == 1.0:
                full_data = r
            if r['fraction'] == 0.1:
                min_data = r

        if full_data and min_data:
            diff = (full_data['test_acc'] - min_data['test_acc']) * 100
            f.write(f"4. Performance Drop (100% vs 10% data): {diff:.2f}%\n\n")

            # add some interpretation
            if diff < 5:
                f.write("   INSIGHT: Model is relatively robust to data reduction.\n")
            elif diff < 15:
                f.write("   INSIGHT: Moderate performance degradation with less data.\n")
            else:
                f.write("   INSIGHT: Significant performance drop with reduced data.\n")

    print(f"Saved summary table to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize data size experiment results")
    parser.add_argument('--metrics',
                       default='experiments/artifacts/data_size_metrics.json',
                       help='path to metrics JSON file from data_size_study.py')
    parser.add_argument('--output-dir',
                       default='experiments/artifacts/plots',
                       help='where to save the plots')
    args = parser.parse_args()

    # load the results
    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        print(f"Error: Metrics file not found at {metrics_path}")
        print("Make sure to run data_size_study.py first!")
        return

    with open(metrics_path) as f:
        results = json.load(f)

    # sort by fraction to make sure plots are in order
    results = sorted(results, key=lambda x: x['fraction'])

    # create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating visualizations from {len(results)} experiments...")
    print(f"Output directory: {output_dir}\n")

    # generate all the plots
    plot_accuracy_vs_data_size(results, output_dir)
    plot_learning_curves(results, output_dir)
    plot_data_efficiency(results, output_dir)
    generate_summary_table(results, output_dir)

    print("\n" + "="*60)
    print("All visualizations generated successfully!")
    print("="*60)
    print(f"\nGenerated files:")
    print(f"  1. {output_dir}/accuracy_vs_data_size.png")
    print(f"  2. {output_dir}/learning_curves.png")
    print(f"  3. {output_dir}/data_efficiency.png")
    print(f"  4. {output_dir}/summary_table.txt")
    print("\nThese plots are ready for your project report!")


if __name__ == '__main__':
    main()
