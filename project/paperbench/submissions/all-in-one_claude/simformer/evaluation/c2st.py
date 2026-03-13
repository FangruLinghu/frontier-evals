"""
Classifier Two-Sample Test (C2ST) implementation.

C2ST is used to evaluate the quality of posterior samples by training
a classifier to distinguish between samples from the true posterior
and samples from the approximate posterior.

A C2ST accuracy of 0.5 indicates perfect samples (indistinguishable),
while 1.0 indicates completely distinguishable samples.

Reference:
- Lopez-Paz & Oquab (2017). Revisiting Classifier Two-Sample Tests.
- Lueckmann et al. (2021). Benchmarking Simulation-Based Inference.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Optional, Tuple, List
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, KFold


def c2st(
    samples_p: torch.Tensor,
    samples_q: torch.Tensor,
    classifier: str = "mlp",
    n_folds: int = 5,
    scoring: str = "accuracy",
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Perform Classifier Two-Sample Test.

    Tests whether two sets of samples come from the same distribution
    by training a classifier to distinguish between them.

    Args:
        samples_p: Samples from first distribution, shape (n_p, dim)
        samples_q: Samples from second distribution, shape (n_q, dim)
        classifier: Type of classifier ("mlp", "knn", "rf")
        n_folds: Number of cross-validation folds
        scoring: Scoring metric
        seed: Random seed

    Returns:
        Tuple of (mean_accuracy, std_accuracy)
    """
    # Convert to numpy
    if isinstance(samples_p, torch.Tensor):
        samples_p = samples_p.detach().cpu().numpy()
    if isinstance(samples_q, torch.Tensor):
        samples_q = samples_q.detach().cpu().numpy()

    # Create labels
    n_p, n_q = len(samples_p), len(samples_q)
    X = np.vstack([samples_p, samples_q])
    y = np.concatenate([np.zeros(n_p), np.ones(n_q)])

    # Shuffle
    if seed is not None:
        np.random.seed(seed)
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    # Create classifier
    if classifier == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(64, 64),
            activation="relu",
            max_iter=500,
            early_stopping=True,
            random_state=seed,
        )
    elif classifier == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors=5)
    elif classifier == "rf":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    else:
        raise ValueError(f"Unknown classifier: {classifier}")

    # Cross-validation
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, X, y, cv=cv, scoring=scoring)

    return float(scores.mean()), float(scores.std())


def c2st_accuracy(
    samples_p: torch.Tensor,
    samples_q: torch.Tensor,
    **kwargs,
) -> float:
    """
    Compute C2ST accuracy (mean only).

    Args:
        samples_p: Samples from first distribution
        samples_q: Samples from second distribution
        **kwargs: Additional arguments passed to c2st

    Returns:
        Mean C2ST accuracy
    """
    mean_acc, _ = c2st(samples_p, samples_q, **kwargs)
    return mean_acc


class NeuralClassifier(nn.Module):
    """Neural network classifier for C2ST with PyTorch."""

    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 64]):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def train_classifier(
    samples_p: torch.Tensor,
    samples_q: torch.Tensor,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    early_stopping_patience: int = 10,
) -> Tuple[NeuralClassifier, List[float]]:
    """
    Train a neural network classifier for C2ST.

    Args:
        samples_p: Samples from first distribution
        samples_q: Samples from second distribution
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device to train on
        early_stopping_patience: Patience for early stopping

    Returns:
        Tuple of (trained_classifier, training_losses)
    """
    if device is None:
        device = samples_p.device

    # Prepare data
    n_p, n_q = len(samples_p), len(samples_q)
    X = torch.cat([samples_p, samples_q], dim=0).to(device)
    y = torch.cat([
        torch.zeros(n_p, 1),
        torch.ones(n_q, 1)
    ]).to(device)

    # Shuffle
    idx = torch.randperm(len(X))
    X, y = X[idx], y[idx]

    # Split train/val
    n_train = int(0.8 * len(X))
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]

    # Create model
    input_dim = X.shape[1]
    model = NeuralClassifier(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # Create dataloader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    losses = []
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        losses.append(epoch_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = criterion(val_logits, y_val).item()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    return model, losses


def c2st_pytorch(
    samples_p: torch.Tensor,
    samples_q: torch.Tensor,
    n_folds: int = 5,
    epochs: int = 100,
    batch_size: int = 128,
    device: Optional[torch.device] = None,
) -> Tuple[float, float]:
    """
    Perform C2ST using PyTorch neural network with cross-validation.

    Args:
        samples_p: Samples from first distribution
        samples_q: Samples from second distribution
        n_folds: Number of cross-validation folds
        epochs: Training epochs per fold
        batch_size: Batch size
        device: Device to use

    Returns:
        Tuple of (mean_accuracy, std_accuracy)
    """
    if device is None:
        device = samples_p.device

    # Prepare data
    n_p, n_q = len(samples_p), len(samples_q)
    X = torch.cat([samples_p, samples_q], dim=0).to(device)
    y = torch.cat([
        torch.zeros(n_p),
        torch.ones(n_q)
    ]).to(device)

    # Shuffle
    idx = torch.randperm(len(X))
    X, y = X[idx], y[idx]

    # K-fold cross-validation
    fold_size = len(X) // n_folds
    accuracies = []

    for fold in range(n_folds):
        # Split data
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size

        val_mask = torch.zeros(len(X), dtype=torch.bool)
        val_mask[val_start:val_end] = True

        X_train, y_train = X[~val_mask], y[~val_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        # Train classifier
        model = NeuralClassifier(X.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()

        train_dataset = TensorDataset(X_train, y_train.unsqueeze(1))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val).squeeze()
            val_preds = (torch.sigmoid(val_logits) > 0.5).float()
            accuracy = (val_preds == y_val).float().mean().item()
            accuracies.append(accuracy)

    return float(np.mean(accuracies)), float(np.std(accuracies))


def c2st_permutation_test(
    samples_p: torch.Tensor,
    samples_q: torch.Tensor,
    n_permutations: int = 100,
    **kwargs,
) -> Tuple[float, float]:
    """
    Perform C2ST with permutation test for p-value.

    Args:
        samples_p: Samples from first distribution
        samples_q: Samples from second distribution
        n_permutations: Number of permutations
        **kwargs: Additional arguments for c2st

    Returns:
        Tuple of (c2st_accuracy, p_value)
    """
    # Observed accuracy
    obs_acc, _ = c2st(samples_p, samples_q, **kwargs)

    # Permutation test
    all_samples = torch.cat([samples_p, samples_q], dim=0)
    n_p = len(samples_p)

    null_accs = []
    for _ in range(n_permutations):
        idx = torch.randperm(len(all_samples))
        perm_p = all_samples[idx[:n_p]]
        perm_q = all_samples[idx[n_p:]]
        null_acc, _ = c2st(perm_p, perm_q, **kwargs)
        null_accs.append(null_acc)

    # P-value: proportion of null accuracies >= observed
    p_value = np.mean([acc >= obs_acc for acc in null_accs])

    return obs_acc, p_value
