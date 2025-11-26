"""
Data size experiment for ASL classifier

This script trains the model on different amounts of data to see how
performance changes. Trying 100%, 50%, 25%, and 10% of the training set.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.model import ASLClassifier


# Custom dataset class - similar to the one in src/dataset.py but works with DataFrames
class SignLanguageFrameDataset(Dataset):
    def __init__(self, df, transform=None):
        # first column is labels, rest are pixels
        self.labels = df.iloc[:, 0].to_numpy()
        self.pixels = df.iloc[:, 1:].to_numpy()
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = int(self.labels[idx])
        # reshape flat pixels back to 28x28 image
        image = self.pixels[idx].reshape(28, 28).astype(np.uint8)

        if self.transform:
            image = self.transform(image)

        return image, label


def set_seeds(seed):
    # make experiments reproducible
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_subset(df, fraction, seed):
    """
    Take a fraction of the dataset while keeping class balance.
    For example, if we have 1000 samples of 'A' and take 50%, we get 500 'A' samples.
    """
    label_col = df.columns[0]
    pieces = []

    # go through each class and sample the same fraction
    for label, group in df.groupby(label_col):
        num_samples = int(len(group) * fraction)
        if num_samples < 1:
            num_samples = 1  # always keep at least one sample per class
        if num_samples > len(group):
            num_samples = len(group)

        sampled = group.sample(n=num_samples, random_state=seed, replace=False)
        pieces.append(sampled)

    # combine all the pieces and shuffle
    subset = pd.concat(pieces)
    subset = subset.sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)
    return subset


def stratified_train_val_split(df, val_ratio, seed):
    """
    Split data into train and validation sets while keeping classes balanced
    """
    label_col = df.columns[0]
    train_parts = []
    val_parts = []

    for label, group in df.groupby(label_col):
        val_size = int(len(group) * val_ratio)

        # edge case handling - make sure we don't take too much or too little
        if len(group) > 1:
            val_size = max(1, min(len(group) - 1, val_size))
        else:
            val_size = 0

        if val_size > 0:
            val_split = group.sample(n=val_size, random_state=seed, replace=False)
            train_split = group.drop(val_split.index)
            val_parts.append(val_split)
        else:
            val_split = group.iloc[0:0]  # empty dataframe
            train_split = group

        train_parts.append(train_split)

    # concat and shuffle everything
    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)

    if len(val_parts) > 0:
        val_df = pd.concat(val_parts).sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)
    else:
        val_df = pd.DataFrame(columns=df.columns)

    return train_df, val_df


def evaluate(model, loader, device):
    """Calculate accuracy on a dataset"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total == 0:
        return 0.0
    return correct / total


def train_one_fraction(fraction, train_df, val_df, test_df, transform, device, epochs, batch_size, lr):
    """
    Train a model on a specific fraction of data and return results
    """
    # create datasets
    train_dataset = SignLanguageFrameDataset(train_df, transform=transform)
    test_dataset = SignLanguageFrameDataset(test_df, transform=transform)

    val_dataset = None
    val_loader = None
    if len(val_df) > 0:
        val_dataset = SignLanguageFrameDataset(val_df, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # initialize model
    model = ASLClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # track best model
    best_state = None
    best_val_acc = -1.0
    history = []

    # training loop
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

        # calculate average loss for this epoch
        train_loss = running_loss / len(train_loader.dataset)

        # evaluate on validation set if we have one
        val_acc = None
        if val_loader is not None:
            val_acc = evaluate(model, val_loader, device)
            # save best model based on validation accuracy
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_state = model.state_dict()

        # save epoch stats
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_acc": val_acc
        })

        # print progress
        val_str = f"{val_acc:.4f}" if val_acc is not None else "n/a"
        print(f"[{int(fraction*100)}%] Epoch {epoch}/{epochs} - loss {train_loss:.4f} - val_acc {val_str}")

    # load best model if we saved one
    if best_state is not None:
        model.load_state_dict(best_state)

    # final evaluation on test set
    test_acc = evaluate(model, test_loader, device)

    # return all the results
    results = {
        "fraction": fraction,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "best_val_acc": None if best_val_acc < 0 else best_val_acc,
        "test_acc": test_acc,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "history": history,
        "state_dict": model.state_dict(),
    }

    return results


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description="Run data size experiments")
    parser.add_argument("--train-csv", default="data/sign_mnist_train.csv", help="path to training CSV")
    parser.add_argument("--test-csv", default="data/sign_mnist_test.csv", help="path to test CSV")
    parser.add_argument("--fractions", default="1.0,0.5,0.25,0.1",
                       help="comma-separated fractions to test (e.g., 1.0,0.5,0.25)")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                       help="what fraction of training data to use for validation")
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument("--output-dir", default="experiments/artifacts",
                       help="where to save results")
    parser.add_argument("--save-weights", action="store_true",
                       help="save model weights for each fraction")

    args = parser.parse_args()

    # set random seeds for reproducibility
    set_seeds(args.seed)

    # parse fractions from string
    fractions = [float(x.strip()) for x in args.fractions.split(",") if x.strip()]

    # create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_dir = output_dir / "weights"
    if args.save_weights:
        weights_dir.mkdir(parents=True, exist_ok=True)

    # check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load full dataset
    print(f"Loading data from {args.train_csv}...")
    train_df_full = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    # basic transform - just convert to tensor
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    # run experiments for each fraction
    results = []
    for frac in fractions:
        print(f"\n{'='*60}")
        print(f"Training with {int(frac*100)}% of data")
        print(f"{'='*60}")

        # create stratified subset
        subset = stratified_subset(train_df_full, frac, seed=args.seed)

        # split into train and validation
        train_df, val_df = stratified_train_val_split(subset, val_ratio=args.val_ratio, seed=args.seed)

        # train the model
        result = train_one_fraction(
            fraction=frac,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            transform=transform,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )

        # save model weights if requested
        state_dict = result.pop("state_dict")
        if args.save_weights:
            weight_path = weights_dir / f"asl_model_frac_{int(frac * 100)}.pth"
            torch.save(state_dict, weight_path)
            print(f"Saved weights to {weight_path}")

        results.append(result)

    # save all metrics to JSON
    metrics_path = output_dir / "data_size_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Wrote metrics to {metrics_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
