"""
Augmentation experiment - testing different data augmentation strategies

This compares how different augmentations (rotation, brightness, etc.)
affect model performance. Could help Member 1 with their augmentation task.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np

from src.model import ASLClassifier
from data_size_study import SignLanguageFrameDataset, stratified_train_val_split, evaluate, set_seeds


# Different augmentation strategies to test
# NOTE: keeping "baseline" with no augmentation for comparison
AUGMENTATION_CONFIGS = {
    'baseline': transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ]),

    'rotation': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(degrees=15),  # rotate up to 15 degrees
        transforms.ToTensor()
    ]),

    'affine': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomAffine(
            degrees=15,
            translate=(0.1, 0.1),  # shift by up to 10%
            scale=(0.9, 1.1)       # zoom 90%-110%
        ),
        transforms.ToTensor()
    ]),

    'brightness': transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor()
    ]),

    'combined': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ]),
}


def train_with_augmentation(aug_name, transform_train, transform_val,
                           train_df, val_df, test_df,
                           device, epochs, batch_size, lr):
    """
    Train a model with a specific augmentation strategy
    """

    # create datasets - use augmentation for training only
    train_dataset = SignLanguageFrameDataset(train_df, transform=transform_train)
    val_dataset = SignLanguageFrameDataset(val_df, transform=transform_val)
    test_dataset = SignLanguageFrameDataset(test_df, transform=transform_val)

    # data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # setup model
    model = ASLClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # training tracking
    best_state = None
    best_val_acc = -1.0
    history = []

    print(f"\nTraining with {aug_name} augmentation...")

    for epoch in range(1, epochs + 1):
        # training
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # validation
        val_acc = evaluate(model, val_loader, device)

        # save best model
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_acc": val_acc
        })

        print(f"  Epoch {epoch}/{epochs} - loss: {train_loss:.4f} - val_acc: {val_acc:.4f}")

    # load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # final test
    test_acc = evaluate(model, test_loader, device)

    return {
        "augmentation": aug_name,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "final_train_loss": history[-1]["train_loss"],
        "epochs": epochs,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare augmentation strategies")
    parser.add_argument("--train-csv", default="data/sign_mnist_train.csv")
    parser.add_argument("--test-csv", default="data/sign_mnist_test.csv")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                       help="fraction of training data for validation")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="experiments/artifacts")

    args = parser.parse_args()

    set_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load data
    train_df_full = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    # split into train and validation
    train_df, val_df = stratified_train_val_split(
        train_df_full, val_ratio=args.val_ratio, seed=args.seed
    )

    print(f"\nDataset sizes:")
    print(f"  Training: {len(train_df)}")
    print(f"  Validation: {len(val_df)}")
    print(f"  Test: {len(test_df)}")

    # validation/test transform - no augmentation
    transform_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    results = []

    # test each augmentation strategy
    for aug_name, transform_train in AUGMENTATION_CONFIGS.items():
        result = train_with_augmentation(
            aug_name=aug_name,
            transform_train=transform_train,
            transform_val=transform_val,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )
        results.append(result)

        print(f"\n{aug_name.upper()} Results:")
        print(f"  Best Val Accuracy: {result['best_val_acc']:.4f}")
        print(f"  Test Accuracy: {result['test_acc']:.4f}")

    # save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "augmentation_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to {metrics_path}")
    print(f"{'='*60}")

    # print comparison table
    print("\nCOMPARISON TABLE:")
    print(f"{'Augmentation':<15} {'Val Acc':<12} {'Test Acc':<12} {'Improvement':<12}")
    print("-" * 60)

    # find baseline for comparison
    baseline_acc = None
    for r in results:
        if r['augmentation'] == 'baseline':
            baseline_acc = r['test_acc']
            break

    # print results sorted by test accuracy
    for r in sorted(results, key=lambda x: x['test_acc'], reverse=True):
        improvement = 0
        if baseline_acc:
            improvement = (r['test_acc'] - baseline_acc) * 100

        print(f"{r['augmentation']:<15} {r['best_val_acc']:<12.4f} "
              f"{r['test_acc']:<12.4f} {improvement:+.2f}%")


if __name__ == "__main__":
    main()
