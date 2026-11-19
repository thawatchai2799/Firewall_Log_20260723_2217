"""
imbalance_methods.py

Five techniques for handling extreme class imbalance, in the main experiment (RQ1):
  1. baseline_smote        - SMOTE + standard cross-entropy (the original baseline)
  2. cost_sensitive        - class-weighted cross-entropy
  3. focal_loss            - focal loss (tunable gamma, alpha)
  4. hybrid_resample       - SMOTE-Tomek or ADASYN + cost-sensitive loss
  5. gan_oversample        - Conditional GAN/VAE oversampling (stub, not yet implemented)

Every function returns (X_resampled, y_resampled, loss_fn) or (X, y, loss_fn) as appropriate,
so that pipeline.py can call every technique through one uniform interface.
"""

from typing import Tuple, Callable
import numpy as np
import torch
import torch.nn as nn

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler

from model import FocalLoss, compute_class_weights


def cap_majority_classes(X: np.ndarray, y: np.ndarray, max_class_size: int = 100_000,
                          random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Undersample only classes larger than max_class_size down to max_class_size before Tomek-link
    cleaning (classes already smaller than this are left untouched).

    Why this is needed: the Tomek-link step in SMOTETomek searches for overlapping nearest-neighbor
    pairs across the whole dataset, including the majority class (not just the minority classes SMOTE
    creates). On real data with millions of rows (e.g. CIC-IDS2017, BENIGN ~2M rows per train
    fold) this step becomes so slow it barely progresses (visible as a single unit stalling in the
    progress log). Undersampling the majority class first lets Tomek-link run on a manageable size
    while fully preserving the "hybrid" idea (oversample minority + clean the boundary).
    """
    classes, counts = np.unique(y, return_counts=True)
    strategy = {int(c): max_class_size for c, cnt in zip(classes, counts) if cnt > max_class_size}
    if not strategy:
        return X, y  # no class exceeds the cap, nothing to do
    rus = RandomUnderSampler(sampling_strategy=strategy, random_state=random_state)
    return rus.fit_resample(X, y)


def compute_capped_sampling_strategy(y_train: np.ndarray, ratio_cap: float = 20.0,
                                      absolute_cap: int = 50_000) -> dict:
    """Compute a capped sampling_strategy instead of SMOTE's default ('auto'), which would try
    to oversample every minority class up to exactly the majority-class size.

    The problem with 'auto' on severely imbalanced real data (e.g. CIC-IDS2017 with BENIGN
    ~2.27M rows versus Heartbleed at just 11 rows = ratio ~206,000:1) is that SMOTE would try to
    synthesise ~2.27M Heartbleed samples from just 11 originals, which:
      1. is statistically unsound (generated from a few neighbours repeated hundreds of thousands of times);
      2. inflates the resampled dataset enormously (all classes together could reach tens of millions of rows),
         consuming RAM/VRAM and training time until it cannot finish in practice.

    This approach caps in two ways: (1) each minority class is oversampled by at most ratio_cap times
    its original count (preventing expansion from too small a base), and (2) never exceeds
    absolute_cap rows per class under any circumstances (preventing dataset blow-up beyond RAM/VRAM).
    Classes already above the cap are left untouched (SMOTE only oversamples, never reduces).
    """
    classes, counts = np.unique(y_train, return_counts=True)
    majority_count = counts.max()
    strategy = {}
    for cls, count in zip(classes, counts):
        target = min(count * ratio_cap, absolute_cap, majority_count)
        if target > count:  # only list classes that will actually be oversampled (target > original)
            strategy[int(cls)] = int(target)
    return strategy


def _safe_smote(X_train, y_train, random_state=42, **smote_kwargs):
    strategy = compute_capped_sampling_strategy(y_train)
    if not strategy:
        # no class needs oversampling (already balanced, or all classes already above the cap)
        return X_train, y_train
    # k_neighbors must not exceed (smallest oversampled class count - 1)
    min_class_count = min(np.sum(y_train == cls) for cls in strategy)
    k_neighbors = max(1, min(5, min_class_count - 1))
    sm = SMOTE(sampling_strategy=strategy, random_state=random_state, k_neighbors=k_neighbors, **smote_kwargs)
    return sm.fit_resample(X_train, y_train)


def baseline_smote(X_train: np.ndarray, y_train: np.ndarray, n_classes: int,
                    random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, Callable]:
    """The original technique (from the source paper) - the primary baseline to beat.
    Uses the capped sampling strategy rather than 'auto' to prevent over-oversampling on data
    with extreme imbalance (see the compute_capped_sampling_strategy docstring)."""
    X_res, y_res = _safe_smote(X_train, y_train, random_state=random_state)
    loss_fn = nn.CrossEntropyLoss()
    return X_res, y_res, loss_fn


def cost_sensitive(X_train: np.ndarray, y_train: np.ndarray, n_classes: int,
                    random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, Callable]:
    """Does not resample; instead reweights the loss to favour minority classes."""
    y_tensor = torch.as_tensor(y_train, dtype=torch.long)
    weights = compute_class_weights(y_tensor, n_classes)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    return X_train, y_train, loss_fn


def focal_loss_method(X_train: np.ndarray, y_train: np.ndarray, n_classes: int,
                       gamma: float = 2.0, use_alpha: bool = True,
                       random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, Callable]:
    y_tensor = torch.as_tensor(y_train, dtype=torch.long)
    alpha = compute_class_weights(y_tensor, n_classes) if use_alpha else None
    loss_fn = FocalLoss(gamma=gamma, alpha=alpha)
    return X_train, y_train, loss_fn


def hybrid_resample(X_train: np.ndarray, y_train: np.ndarray, n_classes: int,
                     method: str = "smote_tomek",
                     random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, Callable]:
    """SMOTE-Tomek (oversample + clean overlap with Tomek links) or ADASYN
    (generates more samples where the classifier struggles) combined with cost-sensitive loss.
    Uses the same capped sampling strategy as baseline_smote to prevent dataset blow-up,
    and always caps the majority class first to prevent Tomek-link/ADASYN from stalling on real data
    with millions of rows (see cap_majority_classes)."""
    # always cap the majority class first, whichever method is chosen (fixes the speed issue at its source)
    X_train, y_train = cap_majority_classes(X_train, y_train, max_class_size=100_000,
                                             random_state=random_state)

    strategy = compute_capped_sampling_strategy(y_train)
    if not strategy:
        X_res, y_res = X_train, y_train
    elif method == "smote_tomek":
        min_class_count = min(np.sum(y_train == cls) for cls in strategy)
        k_neighbors = max(1, min(5, min_class_count - 1))
        sampler = SMOTETomek(sampling_strategy=strategy, random_state=random_state,
                              smote=SMOTE(sampling_strategy=strategy, random_state=random_state,
                                          k_neighbors=k_neighbors))
        X_res, y_res = sampler.fit_resample(X_train, y_train)
    elif method == "adasyn":
        min_class_count = min(np.sum(y_train == cls) for cls in strategy)
        n_neighbors = max(1, min(5, min_class_count - 1))
        sampler = ADASYN(sampling_strategy=strategy, random_state=random_state, n_neighbors=n_neighbors)
        X_res, y_res = sampler.fit_resample(X_train, y_train)
    else:
        raise ValueError(f"unknown method={method}")

    y_tensor = torch.as_tensor(y_res, dtype=torch.long)
    weights = compute_class_weights(y_tensor, n_classes)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    return X_res, y_res, loss_fn


class GANOversampler:
    """STUB - scaffold prepared for future development per the month 5-6 timeline.

    Suggested approach: use the `ctgan` or `sdv` (Synthetic Data Vault) library to train a Conditional
    GAN on minority-class rows only, then sample synthetic examples from it.
    Advantage over geometric SMOTE: better captures the non-linear joint distribution of features.
    Example extension (pseudo-code):

        from ctgan import CTGAN
        minority_df = train_df[train_df[label_col] == minority_class]
        ctgan = CTGAN(epochs=300)
        ctgan.fit(minority_df, discrete_columns=[])
        synthetic = ctgan.sample(n_samples_needed)
    """

    def __init__(self, epochs: int = 300):
        self.epochs = epochs
        raise NotImplementedError(
            "GANOversampler is not fully implemented yet - install `ctgan` and extend it "
            "per the docstring above before real use in months 5-6 of the plan."
        )


TECHNIQUES = {
    "baseline_smote": baseline_smote,
    "cost_sensitive": cost_sensitive,
    "focal_loss": focal_loss_method,
    "hybrid_resample": hybrid_resample,
    # "gan_oversample": GANOversampler  # enable once implemented
}
