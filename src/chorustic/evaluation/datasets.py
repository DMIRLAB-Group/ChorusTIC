from __future__ import annotations

import numpy as np


def remap_labels(y_train: np.ndarray, y_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map labels to contiguous integer ids based on the context split."""
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    classes = np.unique(y_train)
    cls_to_id = {c: i for i, c in enumerate(classes.tolist())}

    y_train_m = np.vectorize(cls_to_id.get)(y_train)
    y_test_m = np.vectorize(cls_to_id.get)(y_test)

    if np.any(y_test_m == None):  # noqa: E711
        missing = set(np.unique(y_test)) - set(classes)
        raise ValueError(f"Test labels contain unseen classes: {sorted(missing)}")

    return y_train_m.astype(np.int64), y_test_m.astype(np.int64), classes


def ensure_2d_timeseries(X: np.ndarray) -> np.ndarray:
    """Coerce UCR-style inputs into ``(N, L)``."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 2:
        return X
    if X.ndim == 3 and X.shape[1] == 1:
        return X[:, 0, :]
    raise ValueError(f"Unexpected univariate time-series shape: {X.shape}")


def ensure_3d_timeseries(X: np.ndarray) -> np.ndarray:
    """Coerce multivariate inputs into ``(N, C, L)``."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        return X[None, None, :]
    if X.ndim == 2:
        return X[:, None, :]
    if X.ndim == 3:
        return X
    raise ValueError(f"Unexpected multivariate time-series shape: {X.shape}")


class VarianceBasedSelector:
    """Select the channels with the largest training-split variance."""

    def __init__(self, new_num_channels: int):
        self.new_num_channels = int(new_num_channels)
        self.support_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> np.ndarray:
        x_transposed = np.swapaxes(x, 1, 2)
        num_samples, seq_len, num_channels = x_transposed.shape
        x_2d = x_transposed.reshape(num_samples * seq_len, num_channels)
        variances = np.var(x_2d, axis=0)
        self.support_ = np.argsort(variances)[::-1][: self.new_num_channels]
        return self.support_

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.support_ is None:
            raise RuntimeError("Call fit before transform.")
        return x[:, self.support_, :]


def crop_pad_per_channel(
    X: np.ndarray,
    *,
    target_len: int,
    seed: int,
    mode: str = "center",
) -> np.ndarray:
    """Crop/pad each channel independently to the signal encoder input length."""
    X3 = ensure_3d_timeseries(X)
    target_len = int(target_len)
    if target_len <= 0:
        raise ValueError(f"target_len must be > 0, got {target_len}")

    N, C, Ltot = X3.shape
    if Ltot == target_len:
        return X3
    if Ltot < target_len:
        pad = np.zeros((N, C, target_len - Ltot), dtype=np.float32)
        return np.concatenate([X3, pad], axis=2)

    if mode not in {"center", "random"}:
        raise ValueError(f"mode must be 'center' or 'random', got {mode}")
    if mode == "center":
        start = (Ltot - target_len) // 2
        return X3[:, :, start : start + target_len]

    rng = np.random.RandomState(int(seed))
    starts = rng.randint(0, Ltot - target_len + 1, size=N)
    out = np.empty((N, C, target_len), dtype=np.float32)
    for i, s in enumerate(starts.tolist()):
        out[i] = X3[i, :, s : s + target_len]
    return out


def concat_channels_and_sample(
    X: np.ndarray,
    *,
    target_len: int,
    seed: int,
    mode: str = "center",
) -> np.ndarray:
    """Flatten channels into one long sequence, then crop/pad to ``target_len``."""
    X = np.asarray(X, dtype=np.float32)
    target_len = int(target_len)
    if target_len <= 0:
        raise ValueError(f"target_len must be > 0, got {target_len}")

    if X.ndim == 2:
        X_flat = X
    elif X.ndim == 3:
        N, C, L = X.shape
        X_flat = X.reshape(N, C * L)
    else:
        raise ValueError(f"Unexpected input shape: {X.shape}")

    N, Ltot = X_flat.shape
    if Ltot == target_len:
        return X_flat
    if Ltot < target_len:
        pad = np.zeros((N, target_len - Ltot), dtype=np.float32)
        return np.concatenate([X_flat, pad], axis=1)

    if mode not in {"center", "random"}:
        raise ValueError(f"mode must be 'center' or 'random', got {mode}")
    if mode == "center":
        start = (Ltot - target_len) // 2
        return X_flat[:, start : start + target_len]

    rng = np.random.RandomState(int(seed))
    starts = rng.randint(0, Ltot - target_len + 1, size=N)
    out = np.empty((N, target_len), dtype=np.float32)
    for i, s in enumerate(starts.tolist()):
        out[i] = X_flat[i, s : s + target_len]
    return out


def maybe_select_channels(
    X_train_raw: np.ndarray,
    X_test_raw: np.ndarray,
    *,
    enabled: bool,
    new_num_channels: int | None,
    dataset_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Optionally select UEA channels by training-split variance."""
    if not enabled or new_num_channels is None:
        return X_train_raw, X_test_raw

    X_train_np = ensure_3d_timeseries(X_train_raw)
    X_test_np = ensure_3d_timeseries(X_test_raw)
    _, c_train, _ = X_train_np.shape
    if c_train <= 1:
        return X_train_np, X_test_np

    k = max(1, min(int(new_num_channels), c_train))
    if k == c_train:
        return X_train_np, X_test_np

    if dataset_name is not None:
        print(f"[VarSelector][UEA] {dataset_name}: channels {c_train} -> {k}")

    selector = VarianceBasedSelector(k)
    selector.fit(X_train_np)
    return selector.transform(X_train_np), selector.transform(X_test_np)


def dataset_names(reader, suite: str, dataset: str | None) -> list[tuple[str, str]]:
    if dataset:
        return [(suite, dataset)]
    if suite == "both":
        return [("ucr", name) for name in reader.dataset_list_ucr] + [("uea", name) for name in reader.dataset_list_uea]
    if suite == "uea":
        return [("uea", name) for name in reader.dataset_list_uea]
    return [("ucr", name) for name in reader.dataset_list_ucr]


def prepare_ucr_uea_arrays(args, suite: str, name: str, reader, input_length: int):
    X_tr, y_tr = reader.read_dataset(name, which_set="train")
    X_te, y_te = reader.read_dataset(name, which_set="test")

    if suite == "uea":
        X_tr, X_te = maybe_select_channels(
            X_tr,
            X_te,
            enabled=bool(args.uea_use_var_selector),
            new_num_channels=(None if args.uea_var_num_channels is None else int(args.uea_var_num_channels)),
            dataset_name=name,
        )
        if args.uea_fusion == "sum_embed":
            X_tr_eval = crop_pad_per_channel(
                X_tr,
                target_len=input_length,
                seed=int(args.seed),
                mode=str(args.uea_mode),
            )
            X_te_eval = crop_pad_per_channel(
                X_te,
                target_len=input_length,
                seed=int(args.seed) + 1,
                mode=str(args.uea_mode),
            )
        else:
            X_tr_eval = concat_channels_and_sample(
                X_tr,
                target_len=input_length,
                seed=int(args.seed),
                mode=str(args.uea_mode),
            )
            X_te_eval = concat_channels_and_sample(
                X_te,
                target_len=input_length,
                seed=int(args.seed) + 1,
                mode=str(args.uea_mode),
            )
    elif suite == "ucr":
        X_tr_eval = ensure_2d_timeseries(X_tr)
        X_te_eval = ensure_2d_timeseries(X_te)
    else:
        raise ValueError(f"Unsupported suite: {suite}")

    y_tr_m, y_te_m, classes = remap_labels(y_tr, y_te)
    return X_tr_eval, y_tr_m, X_te_eval, y_te_m, classes
