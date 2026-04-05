"""
Learned routing policy — lightweight MLP trained on dev set oracle labels.

This is Phase 4 work. The MLP takes 6 uncertainty features and predicts
one of 3 routing modes (A/B/C). It's meant to outperform the rule-based
policy by learning non-linear feature interactions.

Training data comes from running the LID pipeline on the FLEURS dev set
and assigning oracle labels:
  - If top-1 is correct → label = A (single decode would have worked)
  - If correct lang is in top-3 → label = B (multi-hypo would recover it)
  - Otherwise → label = C (fallback needed)
"""
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path

from src.utils import UncertaintySignals, get_logger
from src.routing.policy_rules import RoutingMode, RoutingDecision

log = get_logger("routing.policy_learned")


class LearnedRoutingPolicy:
    """MLP routing policy. Train → save → load → use."""

    def __init__(self, input_dim: int = 11, hidden_dim: int = 64, num_modes: int = 3):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_modes = num_modes
        self._model = None
        self._mode_labels = [RoutingMode.SINGLE, RoutingMode.MULTI_HYPOTHESIS, RoutingMode.FALLBACK]

    def _build_model(self):
        import torch.nn as nn
        self._model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, self.num_modes),
        )
        return self._model

    def train_policy(self, X: np.ndarray, y: np.ndarray,
                     epochs: int = 100, lr: float = 0.001,
                     batch_size: int = 64,
                     val_split: float = 0.15) -> dict:
        """Train the MLP on (uncertainty_features, oracle_label) pairs.
        
        Returns dict with training history (loss, accuracy per epoch).
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Train/val split
        n = len(X)
        indices = np.random.permutation(n)
        val_n = int(n * val_split)
        val_idx, train_idx = indices[:val_n], indices[val_n:]

        X_train = torch.from_numpy(X[train_idx]).float()
        y_train = torch.from_numpy(y[train_idx]).long()
        X_val = torch.from_numpy(X[val_idx]).float()
        y_val = torch.from_numpy(y[val_idx]).long()

        train_ds = TensorDataset(X_train, y_train)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        model = self._build_model().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        # Use class weights to handle imbalanced routing modes
        class_counts = np.bincount(y, minlength=self.num_modes).astype(np.float32)
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * self.num_modes
        criterion = nn.CrossEntropyLoss(
            weight=torch.from_numpy(class_weights).float().to(device)
        )

        history = {"train_loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            epoch_loss /= len(train_idx)

            # Validation
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val.to(device))
                val_loss = criterion(val_logits, y_val.to(device)).item()
                val_preds = val_logits.argmax(dim=-1).cpu().numpy()
                val_acc = (val_preds == y[val_idx]).mean()

            scheduler.step()

            history["train_loss"].append(epoch_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if (epoch + 1) % 10 == 0:
                log.info(f"Epoch {epoch+1}/{epochs}: "
                         f"train_loss={epoch_loss:.4f}, "
                         f"val_loss={val_loss:.4f}, "
                         f"val_acc={val_acc:.3f}")

        self._model = model.cpu()
        return history

    def save(self, path: str | Path):
        import torch
        torch.save({
            "state_dict": self._model.state_dict(),
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_modes": self.num_modes,
        }, path)
        log.info(f"Learned policy saved to {path}")

    def load(self, path: str | Path):
        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.input_dim = ckpt["input_dim"]
        self.hidden_dim = ckpt["hidden_dim"]
        self.num_modes = ckpt["num_modes"]
        self._build_model()
        self._model.load_state_dict(ckpt["state_dict"])
        self._model.eval()
        log.info(f"Learned policy loaded from {path}")

    def decide(self, fused_probs: Dict[str, float],
               uncertainty: UncertaintySignals) -> RoutingDecision:
        """Predict routing mode using the trained MLP."""
        import torch

        if self._model is None:
            raise RuntimeError("Learned policy not trained/loaded. "
                             "Call train_policy() or load() first.")

        # Build 11-dim feature vector: 6 uncertainty + 5 top fused_probs
        uncertainty_vec = uncertainty.to_vector()
        sorted_probs = sorted(fused_probs.values(), reverse=True)[:5]
        sorted_probs = sorted_probs + [0.0] * (5 - len(sorted_probs))
        feature_vec = np.concatenate([uncertainty_vec, sorted_probs]).astype(np.float32)
        features = torch.from_numpy(feature_vec).float().unsqueeze(0)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(features)
            probs = torch.softmax(logits, dim=-1)[0]
            mode_idx = probs.argmax().item()

        mode = self._mode_labels[mode_idx]
        sorted_langs = sorted(fused_probs, key=fused_probs.get, reverse=True)

        if mode == RoutingMode.SINGLE:
            candidates = sorted_langs[:1]
        elif mode == RoutingMode.MULTI_HYPOTHESIS:
            candidates = sorted_langs[:3]
        else:
            candidates = sorted_langs[:5]

        return RoutingDecision(
            mode=mode,
            candidate_languages=candidates,
            confidence=probs[mode_idx].item(),
            reason=f"Learned policy (mode probs: {probs.tolist()})"
        )


def generate_oracle_labels(fused_probs_list: list,
                           uncertainty_list: list,
                           true_langs: list,
                           top_k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Generate training data for the learned policy from dev set results.
    
    Args:
        fused_probs_list: list of fused prob dicts from fusion module
        uncertainty_list: list of UncertaintySignals
        true_langs: list of ground-truth canonical language codes
    
    Returns:
        X: (N, 11) feature matrix (6 uncertainty + 5 top probs)
        y: (N,) labels — 0=Mode A, 1=Mode B, 2=Mode C
    """
    X, y = [], []
    for fused_probs, uncertainty, true_lang in zip(
            fused_probs_list, uncertainty_list, true_langs):
        # Get base uncertainty features
        uncertainty_vec = uncertainty.to_vector()
        # Add top-5 raw probabilities from fused_probs
        sorted_probs = sorted(fused_probs.values(), reverse=True)[:5]
        # Pad to 5 if fewer languages
        sorted_probs = sorted_probs + [0.0] * (5 - len(sorted_probs))
        # Concatenate: (6 uncertainty features) + (5 prob features) = 11
        features = np.concatenate([uncertainty_vec, sorted_probs])
        
        sorted_langs = sorted(fused_probs, key=fused_probs.get, reverse=True)

        if sorted_langs and sorted_langs[0] == true_lang:
            label = 0  # Mode A correct
        elif true_lang in sorted_langs[:top_k]:
            label = 1  # Mode B would recover
        else:
            label = 2  # Mode C needed
        X.append(features)
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)
