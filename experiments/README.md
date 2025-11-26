# Experiments Directory

This folder contains all the data size experiments and visualizations for the final project.

## What's in here

- **data_size_study.py** - Main experiment script that trains models on different amounts of data
- **visualize_results.py** - Creates plots from the experiment results
- **augmentation_study.py** - Tests different augmentation strategies (bonus)
- **artifacts/** - Where results and plots get saved

## Running the data size experiment

Basic usage:
```bash
python experiments/data_size_study.py --epochs 5 --save-weights
```

This will train 4 models using 100%, 50%, 25%, and 10% of the training data.

Optional arguments:
- `--fractions` - Which data fractions to test (default: "1.0,0.5,0.25,0.1")
- `--epochs` - Number of training epochs (default: 5)
- `--batch-size` - Batch size for training (default: 64)
- `--val-ratio` - Validation split ratio (default: 0.1)
- `--save-weights` - Save model weights for each fraction

## Creating visualizations

After running the experiment, generate plots with:
```bash
python experiments/visualize_results.py
```

This creates:
1. **accuracy_vs_data_size.png** - Main result showing how accuracy scales with data
2. **learning_curves.png** - Training dynamics for each fraction
3. **data_efficiency.png** - Marginal returns analysis
4. **summary_table.txt** - Text summary of results

All plots are saved to `experiments/artifacts/plots/`

## Testing augmentation strategies (optional)

Compare different augmentation approaches:
```bash
python experiments/augmentation_study.py --epochs 10
```

Tests:
- baseline (no augmentation)
- rotation only
- affine transforms (rotation + translation + scale)
- brightness/contrast adjustment
- combined (all augmentations)

## Results

After running experiments, you'll find:
```
experiments/artifacts/
├── data_size_metrics.json       # Raw experimental data
├── augmentation_metrics.json    # Augmentation results (if run)
├── weights/                     # Saved model checkpoints
│   ├── asl_model_frac_10.pth
│   ├── asl_model_frac_25.pth
│   ├── asl_model_frac_50.pth
│   └── asl_model_frac_100.pth
└── plots/                       # Visualizations
    ├── accuracy_vs_data_size.png
    ├── learning_curves.png
    ├── data_efficiency.png
    └── summary_table.txt
```

## Notes

- Make sure to set PYTHONPATH if you get import errors:
  ```bash
  export PYTHONPATH=/path/to/asl-translator-copy
  ```

- The experiments run on CPU by default but will use GPU if available

- Training on full dataset (100%) takes about 5-10 minutes on CPU for 5 epochs

- All experiments use seed=42 for reproducibility
