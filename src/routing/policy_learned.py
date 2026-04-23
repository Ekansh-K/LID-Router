"""
Learned routing policy — lightweight MLP trained on dev set oracle labels.

Phase 1+2 additions:
  After the MLP makes its A/B/C prediction, a post-decision override layer
  enforces the force_mode_b_threshold and flat-gap guard from confusion_clusters.yaml.
  This means even a confident MLP prediction of Mode A will be downgraded to
  Mode B for languages like urd (LID=67%) and srp (LID=83%).
"""
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path

from src.utils import UncertaintySignals, get_logger, get_language_map
from src.routing.policy_rules import RoutingMode, RoutingDecision, _apply_temperature
from src.routing.confusion_map import ConfusionMap

log = get_logger("routing.policy_learned")


class LearnedRoutingPolicy:
    """MLP routing policy. Train → save → load → use."""

    def __init__(self, input_dim: int = 11, hidden_dim: int = 64, num_modes: int = 3,
                 confusion_map: Optional[ConfusionMap] = None):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_modes = num_modes
        self._model = None
        self._mode_labels = [RoutingMode.SINGLE, RoutingMode.MULTI_HYPOTHESIS, RoutingMode.FALLBACK]
        # Phase 1+2: post-decision override guards
        self._confusion_map = confusion_map or ConfusionMap()
        self._flat_gap_override = 0.10  # force Mode B when tempered gap < this

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
        import torch.nn as nn
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.input_dim = ckpt["input_dim"]
        self.hidden_dim = ckpt["hidden_dim"]
        self.num_modes = ckpt["num_modes"]
        sd = ckpt["state_dict"]
        # Detect legacy architecture (no BatchNorm): expects keys 0, 3, 6
        # Current arch (with BatchNorm) has keys 0, 1, 4, 5, 8, 9, 12
        if "0.weight" in sd and "1.weight" not in sd and "4.weight" not in sd:
            self._model = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),   # 0
                nn.ReLU(),                                     # 1
                nn.Dropout(0.2),                              # 2
                nn.Linear(self.hidden_dim, self.hidden_dim),  # 3
                nn.ReLU(),                                     # 4
                nn.Dropout(0.2),                              # 5
                nn.Linear(self.hidden_dim, self.num_modes),   # 6
            )
        else:
            self._build_model()
        self._model.load_state_dict(sd)
        self._model.eval()
        log.info(f"Learned policy loaded from {path}")

    def decide(self, fused_probs: Dict[str, float],
               uncertainty: UncertaintySignals) -> RoutingDecision:
        """Predict routing mode using the trained MLP, then apply Phase 1+2 overrides."""
        import torch

        if self._model is None:
            raise RuntimeError("Learned policy not trained/loaded. "
                             "Call train_policy() or load() first.")

        # Build feature vector: 6 uncertainty + up to 5 top fused_probs
        uncertainty_vec = uncertainty.to_vector()
        sorted_probs = sorted(fused_probs.values(), reverse=True)[:5]
        sorted_probs = sorted_probs + [0.0] * (5 - len(sorted_probs))
        feature_vec = np.concatenate([uncertainty_vec, sorted_probs]).astype(np.float32)[:self.input_dim]
        features = torch.from_numpy(feature_vec).float().unsqueeze(0)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(features)
            probs = torch.softmax(logits, dim=-1)[0]
            mode_idx = probs.argmax().item()

        mode = self._mode_labels[mode_idx]

        lang_map = get_language_map()
        sorted_langs = [
            l for l in sorted(fused_probs, key=fused_probs.get, reverse=True)
            if lang_map.asr_capable(l)
        ]

        if mode == RoutingMode.SINGLE:
            candidates = sorted_langs[:1]
        elif mode == RoutingMode.MULTI_HYPOTHESIS:
            candidates = sorted_langs[:3]
        else:
            candidates = sorted_langs[:5]

        mlp_reason = f"Learned policy (mode probs: {[round(p,3) for p in probs.tolist()]})"

        # ── Phase 1+2 post-decision override ─────────────────────────────────
        if sorted_langs:
            top1 = sorted_langs[0]

            # RAW prob for the threshold check (pre-temperature).
            raw_top1_prob = fused_probs.get(top1, 0.0)

            # Phase 1: force_mode_b_threshold — uses RAW prob, not tempered.
            force_b_thresh = self._confusion_map.force_mode_b_threshold(top1)
            if mode == RoutingMode.SINGLE and force_b_thresh > 0 and raw_top1_prob < force_b_thresh:
                candidates = sorted_langs[:3]
                _inject_partners_learned(candidates, top1, self._confusion_map, fused_probs, lang_map)
                mode = RoutingMode.MULTI_HYPOTHESIS
                mlp_reason += (f" | Override: forced Mode B for '{top1}' "
                               f"(raw_prob={raw_top1_prob:.3f} < thresh={force_b_thresh:.3f})")

            # Phase 2: Flat-gap guard — uses temperature-scaled gap.
            elif mode == RoutingMode.SINGLE and self._confusion_map.is_confused(top1):
                tau = self._confusion_map.temperature(top1)
                tempered = _apply_temperature(fused_probs, tau)
                t_sorted = [l for l in sorted(tempered, key=tempered.get, reverse=True)
                            if lang_map.asr_capable(l)]
                t_top2_prob = tempered.get(t_sorted[1], 0.0) if len(t_sorted) > 1 else 0.0
                tempered_gap = tempered.get(t_sorted[0], 0.0) - t_top2_prob
                if tempered_gap < self._flat_gap_override:
                    candidates = sorted_langs[:3]
                    _inject_partners_learned(candidates, top1, self._confusion_map, fused_probs, lang_map)
                    mode = RoutingMode.MULTI_HYPOTHESIS
                    mlp_reason += (f" | Override: flat-gap guard for '{top1}' "
                                   f"(gap={tempered_gap:.3f} < {self._flat_gap_override}, tau={tau:.1f})")

        return RoutingDecision(
            mode=mode,
            candidate_languages=candidates,
            confidence=probs[mode_idx].item(),
            reason=mlp_reason
        )


def _inject_partners_learned(candidates: list, top1: str, confusion_map,
                              fused_probs: dict, lang_map):
    """In-place: add confusion partners to candidate list if not already there.

    Uses fused_probs (raw) to check partner availability.
    Caps at 2 extra partners beyond the initial candidate list (matching rules policy).
    """
    partners = confusion_map.get_partners(top1)
    candidate_set = set(candidates)
    added = 0
    for p in partners:
        if added >= 2:
            break
        if (p not in candidate_set
                and lang_map.asr_capable(p)):
            candidates.append(p)
            candidate_set.add(p)
            added += 1
    # Hard cap: never exceed 5 candidates total
    del candidates[5:]


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
