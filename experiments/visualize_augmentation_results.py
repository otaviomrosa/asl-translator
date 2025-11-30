"""
Visualize augmentation study results

Creates publication-quality plots comparing different augmentation strategies
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


def load_results(metrics_path):
    """Load augmentation experiment results"""
    with open(metrics_path, 'r') as f:
        return json.load(f)


def plot_augmentation_comparison(results, output_dir):
    """Create bar chart comparing augmentation strategies"""

    # Sort by test accuracy
    results_sorted = sorted(results, key=lambda x: x['test_acc'], reverse=True)

    strategies = [r['augmentation'] for r in results_sorted]
    test_accs = [r['test_acc'] * 100 for r in results_sorted]
    val_accs = [r['best_val_acc'] * 100 for r in results_sorted]

    # Find baseline for improvement calculation
    baseline_acc = None
    for r in results:
        if r['augmentation'] == 'baseline':
            baseline_acc = r['test_acc'] * 100
            break

    improvements = [acc - baseline_acc for acc in test_accs]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Test Accuracy Comparison
    x = np.arange(len(strategies))
    width = 0.35

    bars1 = ax1.bar(x - width/2, val_accs, width, label='Validation Acc',
                    color='steelblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, test_accs, width, label='Test Acc',
                    color='coral', alpha=0.8)

    ax1.set_xlabel('Augmentation Strategy')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Augmentation Strategy Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, rotation=15, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim([80, 105])

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=8)

    # Plot 2: Improvement over Baseline
    colors = ['green' if imp >= 0 else 'red' for imp in improvements]
    bars = ax2.bar(strategies, improvements, color=colors, alpha=0.7)

    ax2.set_xlabel('Augmentation Strategy')
    ax2.set_ylabel('Improvement over Baseline (%)')
    ax2.set_title('Performance Gain from Augmentation')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_xticklabels(strategies, rotation=15, ha='right')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{imp:+.2f}%',
                ha='center', va='bottom' if imp >= 0 else 'top',
                fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'augmentation_comparison.png', bbox_inches='tight')
    print(f"Saved: {output_dir / 'augmentation_comparison.png'}")
    plt.close()


def plot_learning_curves(results, output_dir):
    """Plot training curves for each augmentation strategy"""

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, result in enumerate(results):
        if idx >= len(axes):
            break

        ax = axes[idx]
        history = result['history']
        epochs = [h['epoch'] for h in history]
        train_loss = [h['train_loss'] for h in history]
        val_acc = [h['val_acc'] * 100 for h in history]

        # Plot loss on primary y-axis
        color1 = 'tab:red'
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Loss', color=color1)
        line1 = ax.plot(epochs, train_loss, color=color1, linewidth=2,
                       marker='o', markersize=4, label='Train Loss')
        ax.tick_params(axis='y', labelcolor=color1)
        ax.grid(alpha=0.3, linestyle='--')

        # Plot accuracy on secondary y-axis
        ax2 = ax.twinx()
        color2 = 'tab:blue'
        ax2.set_ylabel('Validation Accuracy (%)', color=color2)
        line2 = ax2.plot(epochs, val_acc, color=color2, linewidth=2,
                        marker='s', markersize=4, label='Val Acc')
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_ylim([0, 105])

        # Title with final metrics
        final_test = result['test_acc'] * 100
        ax.set_title(f"{result['augmentation'].upper()}\nTest Acc: {final_test:.2f}%",
                    fontweight='bold')

        # Add legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='center right', fontsize=8)

    # Hide extra subplots
    for idx in range(len(results), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Training Dynamics for Each Augmentation Strategy',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'augmentation_learning_curves.png', bbox_inches='tight')
    print(f"Saved: {output_dir / 'augmentation_learning_curves.png'}")
    plt.close()


def create_summary_table(results, output_dir):
    """Create text summary of results"""

    output_file = output_dir / 'augmentation_summary.txt'

    # Sort by test accuracy
    results_sorted = sorted(results, key=lambda x: x['test_acc'], reverse=True)

    # Find baseline
    baseline_acc = None
    for r in results:
        if r['augmentation'] == 'baseline':
            baseline_acc = r['test_acc'] * 100
            break

    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("AUGMENTATION STUDY RESULTS\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"{'Strategy':<15} {'Val Acc':<12} {'Test Acc':<12} {'Improvement':<15} {'Status'}\n")
        f.write("-" * 80 + "\n")

        for r in results_sorted:
            strategy = r['augmentation']
            val_acc = r['best_val_acc'] * 100
            test_acc = r['test_acc'] * 100
            improvement = test_acc - baseline_acc

            if strategy == 'baseline':
                status = '(baseline)'
            elif test_acc == max(x['test_acc'] for x in results) * 100:
                status = '⭐ BEST'
            elif improvement > 3:
                status = '✓ Excellent'
            elif improvement > 1:
                status = '✓ Good'
            elif improvement > 0:
                status = '~ Marginal'
            else:
                status = '✗ Worse'

            f.write(f"{strategy:<15} {val_acc:>6.2f}%     {test_acc:>6.2f}%     "
                   f"{improvement:>+6.2f}%        {status}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("KEY FINDINGS:\n")
        f.write("=" * 80 + "\n\n")

        best = results_sorted[0]
        f.write(f"1. BEST STRATEGY: {best['augmentation'].upper()}\n")
        f.write(f"   - Test Accuracy: {best['test_acc']*100:.2f}%\n")
        f.write(f"   - Improvement: +{(best['test_acc']*100 - baseline_acc):.2f}%\n\n")

        f.write(f"2. BASELINE (no augmentation): {baseline_acc:.2f}%\n\n")

        f.write("3. RECOMMENDATIONS:\n")
        if best['test_acc'] * 100 > 94:
            f.write(f"   ✓ Use '{best['augmentation']}' augmentation in production\n")
        f.write(f"   ✓ Augmentation provides {(best['test_acc']*100 - baseline_acc):.2f}% accuracy boost\n")
        f.write(f"   ✓ This closes gap toward 95%+ target accuracy\n\n")

        f.write("4. MEMBER 1 (ARCHITECT) ACTION ITEMS:\n")
        f.write(f"   → Update train.py to use '{best['augmentation']}' transforms\n")
        f.write(f"   → Expected final model accuracy: ~{best['test_acc']*100:.2f}%\n")
        f.write(f"   → Code reference: experiments/augmentation_study.py:26-60\n\n")

    print(f"Saved: {output_file}")

    # Also print to console
    with open(output_file, 'r') as f:
        print("\n" + f.read())


def main():
    """Generate all augmentation visualizations"""

    artifacts_dir = Path("experiments/artifacts")
    plots_dir = artifacts_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = artifacts_dir / "augmentation_metrics.json"

    if not metrics_file.exists():
        print(f"Error: {metrics_file} not found!")
        print("Please run: python experiments/augmentation_study.py --epochs 10")
        return

    print("Loading augmentation results...")
    results = load_results(metrics_file)

    print(f"\nFound {len(results)} augmentation strategies\n")

    print("Creating visualizations...")
    plot_augmentation_comparison(results, plots_dir)
    plot_learning_curves(results, plots_dir)
    create_summary_table(results, plots_dir)

    print("\n" + "="*60)
    print("All visualizations created successfully!")
    print("="*60)
    print(f"\nOutput location: {plots_dir}/")
    print("\nGenerated files:")
    print("  - augmentation_comparison.png")
    print("  - augmentation_learning_curves.png")
    print("  - augmentation_summary.txt")


if __name__ == "__main__":
    main()
