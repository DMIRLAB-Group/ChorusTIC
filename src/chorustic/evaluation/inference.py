from __future__ import annotations

import gc
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from chorustic.model import ChorusTIC


def is_cuda_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return (
        ("out of memory" in msg and "cuda" in msg)
        or "cublas_status_alloc_failed" in msg
        or "cuda error: out of memory" in msg
    )


def recover_after_cuda_oom() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def reduce_batch_size(batch_size: int, min_batch_size: int) -> int:
    batch_size = max(1, int(batch_size))
    min_batch_size = max(1, int(min_batch_size))
    if batch_size <= min_batch_size:
        return batch_size
    return max(min_batch_size, batch_size // 2)


def set_signal_encoder_batch_size(model: nn.Module, batch_size: int) -> None:
    target = getattr(model, "module", model)
    pre = getattr(target, "pre_mantis_encoder", None)
    if pre is not None and hasattr(pre, "mantis_batch_size"):
        setattr(pre, "mantis_batch_size", int(batch_size))
    if hasattr(target, "mantis_batch_size"):
        setattr(target, "mantis_batch_size", int(batch_size))


def set_task_level_ensemble_batch_size(model: nn.Module, batch_size: int) -> None:
    target = getattr(model, "module", model)
    if hasattr(target, "task_level_chorus_ensemble_batch_size"):
        setattr(target, "task_level_chorus_ensemble_batch_size", max(1, int(batch_size)))


def _cheap_ts_summary_features(X: np.ndarray) -> np.ndarray:
    X_arr = np.asarray(X, dtype=np.float32)
    if X_arr.ndim == 2:
        X_arr = X_arr[:, None, :]
    elif X_arr.ndim != 3:
        raise ValueError(f"Expected X_train to be 2D or 3D time-series array, got shape {tuple(X_arr.shape)}")

    X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
    flat = X_arr.reshape(X_arr.shape[0], -1)
    diff = np.diff(X_arr, axis=-1)
    diff_flat = diff.reshape(X_arr.shape[0], -1) if diff.shape[-1] > 0 else np.zeros((X_arr.shape[0], 1), dtype=X_arr.dtype)

    win = max(1, int(X_arr.shape[-1]) // 10)
    slope = X_arr[..., -win:].mean(axis=(1, 2)) - X_arr[..., :win].mean(axis=(1, 2))
    channel_mean = X_arr.mean(axis=-1)
    channel_std = X_arr.std(axis=-1)

    features = [
        flat.mean(axis=1),
        flat.std(axis=1),
        flat.min(axis=1),
        flat.max(axis=1),
        np.percentile(flat, 25, axis=1),
        np.percentile(flat, 50, axis=1),
        np.percentile(flat, 75, axis=1),
        np.mean(flat * flat, axis=1),
        diff_flat.mean(axis=1),
        diff_flat.std(axis=1),
        np.abs(diff_flat).mean(axis=1),
        slope,
        channel_mean.mean(axis=1),
        channel_mean.std(axis=1),
        channel_mean.min(axis=1),
        channel_mean.max(axis=1),
        channel_std.mean(axis=1),
        channel_std.std(axis=1),
        channel_std.min(axis=1),
        channel_std.max(axis=1),
    ]
    return np.nan_to_num(np.stack(features, axis=1).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _allocate_context_budgets(
    y_train: np.ndarray,
    *,
    context_size: int,
    allocation: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    classes, counts = np.unique(y_train, return_counts=True)
    n_classes = int(len(classes))
    n_train = int(len(y_train))
    effective_size = int(context_size)
    if effective_size < n_classes:
        print(
            f"[ContextSelect][warning] requested context_size={context_size} is smaller than "
            f"n_classes={n_classes}; using {n_classes} to keep at least one sample per class."
        )
        effective_size = n_classes
    effective_size = min(effective_size, n_train)

    budgets = np.ones(n_classes, dtype=np.int64)
    remaining = int(effective_size - n_classes)
    if remaining <= 0:
        return classes, counts, budgets

    if allocation == "balanced":
        weights = np.ones(n_classes, dtype=np.float64)
    elif allocation == "sqrt":
        weights = np.sqrt(counts.astype(np.float64))
    else:
        raise ValueError(f"Unknown context allocation mode: {allocation}")

    desired_extra = remaining * weights / max(float(weights.sum()), 1e-12)
    capacities = counts.astype(np.int64) - 1
    extra = np.minimum(np.floor(desired_extra).astype(np.int64), capacities)
    budgets += extra

    left = int(effective_size - budgets.sum())
    remainders = desired_extra - np.floor(desired_extra)
    while left > 0:
        available = np.where(budgets < counts)[0]
        if len(available) == 0:
            break
        order = sorted(
            available.tolist(),
            key=lambda i: (-float(remainders[i]), -int(counts[i]), str(classes[i])),
        )
        for i in order:
            if left <= 0:
                break
            if budgets[i] < counts[i]:
                budgets[i] += 1
                left -= 1

    return classes, counts, budgets


def select_context_indices(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    context_mode: str,
    context_size: int,
    seed: int,
    allocation: str = "balanced",
    proto_ratio: float = 0.5,
) -> np.ndarray:
    y_arr = np.asarray(y_train)
    n_train = int(len(y_arr))
    if context_mode == "full" or int(context_size) <= 0 or int(context_size) >= n_train:
        return np.arange(n_train, dtype=np.int64)

    rng = np.random.default_rng(int(seed))
    classes, _counts, budgets = _allocate_context_budgets(
        y_arr,
        context_size=int(context_size),
        allocation=str(allocation),
    )

    selected: list[np.ndarray] = []
    if context_mode == "stratified_random":
        for cls, budget in zip(classes, budgets):
            idx_k = np.where(y_arr == cls)[0]
            if len(idx_k) <= int(budget):
                selected.append(idx_k.astype(np.int64, copy=False))
            else:
                selected.append(rng.choice(idx_k, size=int(budget), replace=False).astype(np.int64, copy=False))
    elif context_mode == "class_proto_kcenter":
        F = _cheap_ts_summary_features(X_train)
        F = (F - F.mean(axis=0, keepdims=True)) / (F.std(axis=0, keepdims=True) + 1e-6)
        F = np.nan_to_num(F.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        ratio = min(max(float(proto_ratio), 0.0), 1.0)
        for cls, budget in zip(classes, budgets):
            idx_k = np.where(y_arr == cls)[0]
            m_k = int(budget)
            if len(idx_k) <= m_k:
                selected.append(idx_k.astype(np.int64, copy=False))
                continue

            F_k = F[idx_k]
            center = F_k.mean(axis=0)
            center_dist = np.sum((F_k - center) ** 2, axis=1)
            proto_order = np.argsort(center_dist, kind="mergesort")
            n_proto = int(round(m_k * ratio))
            n_proto = min(max(1, n_proto), m_k)
            selected_local = proto_order[:n_proto].astype(np.int64).tolist()

            min_sq = np.full(len(idx_k), np.inf, dtype=np.float32)
            for local_idx in selected_local:
                dist = np.sum((F_k - F_k[local_idx]) ** 2, axis=1)
                min_sq = np.minimum(min_sq, dist.astype(np.float32, copy=False))
            min_sq[np.asarray(selected_local, dtype=np.int64)] = -np.inf

            while len(selected_local) < m_k:
                next_local = int(np.argmax(min_sq))
                if not np.isfinite(min_sq[next_local]):
                    break
                selected_local.append(next_local)
                dist = np.sum((F_k - F_k[next_local]) ** 2, axis=1)
                min_sq = np.minimum(min_sq, dist.astype(np.float32, copy=False))
                min_sq[np.asarray(selected_local, dtype=np.int64)] = -np.inf

            selected.append(idx_k[np.asarray(selected_local, dtype=np.int64)].astype(np.int64, copy=False))
    else:
        raise ValueError(f"Unknown classifier_v2 context mode: {context_mode}")

    selected_idx = np.concatenate(selected).astype(np.int64, copy=False) if selected else np.empty(0, dtype=np.int64)
    rng.shuffle(selected_idx)
    return selected_idx


def _random_crop_resize(x: torch.Tensor, crop_rate: float, size: int | None = None) -> torch.Tensor:
    seq_len = int(x.shape[-1])
    size = seq_len if size is None else int(size)
    cropped_seq_len = max(1, int(seq_len * (1.0 - float(crop_rate))))
    start_idx = torch.randint(0, seq_len - cropped_seq_len + 1, (1,), device=x.device).item()
    x_cropped = x[:, :, start_idx : start_idx + cropped_seq_len]
    return F.interpolate(x_cropped, size=size, mode="linear", align_corners=False)


def _softmax(x: np.ndarray, axis: int = -1, temperature: float = 0.9) -> np.ndarray:
    x = x / float(temperature)
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def _make_classifier_v2_members(
    *,
    n_estimators: int,
    n_augmentations: int,
    n_classes: int,
    class_shift: bool,
    crop_rate_range: tuple[float, float],
    seed: int,
) -> list[tuple[int, float]]:
    rng = random.Random(int(seed))
    n_estimators = max(1, int(n_estimators))
    n_augmentations = max(1, int(n_augmentations))

    if bool(class_shift) and n_estimators > 1:
        base_offsets = list(range(int(n_classes)))
        rng.shuffle(base_offsets)
        offsets = [base_offsets[i % len(base_offsets)] for i in range(n_estimators)]
    else:
        offsets = [0 for _ in range(n_estimators)]

    lo, hi = crop_rate_range
    lo = float(lo)
    hi = float(hi)
    if not (0.0 <= lo <= hi < 1.0):
        raise ValueError(f"crop_rate_range must satisfy 0 <= lo <= hi < 1, got {crop_rate_range}")

    members: list[tuple[int, float]] = []
    for off in offsets:
        for _ in range(n_augmentations):
            crop_rate = lo if lo == hi else (lo + (hi - lo) * rng.random())
            members.append((int(off), float(crop_rate)))
    return members


def _predict_classifier_v2_once(
    *,
    model: ChorusTIC,
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_test: np.ndarray,
    device: torch.device,
    n_estimators: int,
    class_shift: bool,
    crop_rate_range: tuple[float, float],
    n_augmentations: int,
    softmax_temperature: float,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    y_context = np.asarray(y_context, dtype=np.int64)
    n_classes = int(np.max(y_context)) + 1 if y_context.size else 0
    if n_classes <= 0:
        raise ValueError("Context labels are empty.")

    X_context = np.asarray(X_context, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    if X_context.ndim not in (2, 3) or X_test.ndim != X_context.ndim:
        raise ValueError(
            "classifier_v2 expects context/test arrays to both be 2D (N,L) "
            f"or 3D (N,C,L), got {X_context.shape} and {X_test.shape}."
        )

    X_all = np.concatenate([X_context, X_test], axis=0)
    train_size = int(y_context.shape[0])
    test_size = int(X_test.shape[0])
    x_all_t = torch.from_numpy(X_all.astype(np.float32)).to(device)
    input_is_multichannel = x_all_t.ndim == 3
    if x_all_t.ndim == 2:
        x_all_t = x_all_t.unsqueeze(1)
    elif x_all_t.ndim != 3:
        raise ValueError(f"Unexpected classifier_v2 input shape: {tuple(x_all_t.shape)}")

    members = _make_classifier_v2_members(
        n_estimators=n_estimators,
        n_augmentations=n_augmentations,
        n_classes=n_classes,
        class_shift=class_shift,
        crop_rate_range=crop_rate_range,
        seed=seed,
    )

    outputs: list[np.ndarray] = []
    offsets: list[int] = []
    bs = max(1, int(batch_size))
    for start in range(0, len(members), bs):
        chunk = members[start : start + bs]
        Xs = []
        ys = []
        for off, crop_rate in chunk:
            x_aug = _random_crop_resize(x_all_t, crop_rate=float(crop_rate))
            if not input_is_multichannel:
                x_aug = x_aug.squeeze(1)
            Xs.append(x_aug)
            y_shift = ((y_context + int(off)) % n_classes).astype(np.float32)
            ys.append(torch.from_numpy(y_shift).to(device))
            offsets.append(int(off))

        X_batch = torch.stack(Xs, dim=0)
        y_batch = torch.stack(ys, dim=0)
        with torch.no_grad():
            out = model(
                X_batch,
                y_batch,
                feature_shuffles=None,
                return_logits=True,
                softmax_temperature=float(softmax_temperature),
                inference_config=None,
            )
        out_np = out.float().cpu().numpy()
        if out_np.shape[1] != test_size:
            raise RuntimeError(f"Unexpected model output shape: {out_np.shape}, expected test_size={test_size}")
        outputs.append(out_np[..., :n_classes])

    outputs_np = np.concatenate(outputs, axis=0)
    avg = None
    for out, off in zip(outputs_np, offsets):
        if off != 0:
            out = np.concatenate([out[..., off:], out[..., :off]], axis=-1)
        avg = out if avg is None else avg + out
    avg = avg / len(offsets)
    proba = _softmax(avg, axis=-1, temperature=float(softmax_temperature))
    proba = proba / proba.sum(axis=1, keepdims=True)
    return np.argmax(proba, axis=1).astype(np.int64)


def predict_classifier_v2_with_oom_retry(
    *,
    custom_model: ChorusTIC,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    device: torch.device,
    args,
    crop_rate_range: tuple[float, float],
    initial_v2_batch_size: int,
    initial_signal_encoder_batch_size: int,
    initial_task_level_chorus_batch_size: int,
) -> tuple[np.ndarray, int, int, int]:
    v2_batch_size = max(1, int(initial_v2_batch_size))
    signal_encoder_batch_size = max(1, int(initial_signal_encoder_batch_size))
    task_level_chorus_batch_size = max(1, int(initial_task_level_chorus_batch_size))
    min_v2_batch_size = max(1, int(args.oom_min_v2_batch_size))
    min_signal_encoder_batch_size = max(1, int(args.oom_min_signal_encoder_batch_size))
    min_task_level_chorus_batch_size = max(1, int(args.oom_min_task_level_chorus_ensemble_batch_size))

    context_seed = int(args.seed if args.v2_context_seed is None else args.v2_context_seed)
    ctx_idx = select_context_indices(
        X_train,
        y_train,
        context_mode=str(args.v2_context_mode),
        context_size=int(args.v2_context_size),
        seed=context_seed,
        allocation=str(args.v2_context_allocation),
        proto_ratio=float(args.v2_context_proto_ratio),
    )
    X_context = X_train[ctx_idx]
    y_context = y_train[ctx_idx]
    if str(args.v2_context_mode) == "full":
        print(f"[ContextSelect][classifier_v2] mode=full selected={len(ctx_idx)}/{len(y_train)}")
    else:
        print(
            f"[ContextSelect][classifier_v2] mode={args.v2_context_mode} "
            f"requested_size={args.v2_context_size} "
            f"selected={len(ctx_idx)}/{len(y_train)} "
            f"classes={len(np.unique(y_train))} "
            f"seed={context_seed}"
        )

    attempt = 1
    while True:
        set_signal_encoder_batch_size(custom_model, signal_encoder_batch_size)
        set_task_level_ensemble_batch_size(custom_model, task_level_chorus_batch_size)
        last_error = None
        try:
            y_pred = _predict_classifier_v2_once(
                model=custom_model,
                X_context=X_context,
                y_context=y_context,
                X_test=X_test,
                device=device,
                n_estimators=int(args.n_estimators),
                class_shift=bool(args.v2_class_shift),
                crop_rate_range=crop_rate_range,
                n_augmentations=int(args.v2_n_augmentations),
                softmax_temperature=float(args.softmax_temperature),
                batch_size=int(v2_batch_size),
                seed=int(args.seed),
            )
            return y_pred, v2_batch_size, signal_encoder_batch_size, task_level_chorus_batch_size
        except Exception as exc:
            if not bool(args.oom_retry) or not is_cuda_oom_error(exc):
                raise
            last_error = str(exc)

        recover_after_cuda_oom()
        next_v2 = reduce_batch_size(v2_batch_size, min_v2_batch_size)
        next_signal_encoder = reduce_batch_size(signal_encoder_batch_size, min_signal_encoder_batch_size)
        next_task_level_chorus = reduce_batch_size(task_level_chorus_batch_size, min_task_level_chorus_batch_size)
        if (
            next_v2 == v2_batch_size
            and next_signal_encoder == signal_encoder_batch_size
            and next_task_level_chorus == task_level_chorus_batch_size
        ):
            raise RuntimeError(
                "classifier_v2 CUDA OOM retry exhausted at "
                f"v2_batch_size={v2_batch_size}, "
                f"signal_encoder_batch_size={signal_encoder_batch_size}, "
                f"task_level_chorus_ensemble_batch_size={task_level_chorus_batch_size}. "
                f"Last error: {last_error}"
            )
        print(
            "[OOM][classifier_v2] CUDA OOM; retrying "
            f"(attempt {attempt + 1}) with v2_batch_size {v2_batch_size}->{next_v2}, "
            f"signal_encoder_batch_size {signal_encoder_batch_size}->{next_signal_encoder}, "
            f"task_level_chorus_ensemble_batch_size {task_level_chorus_batch_size}->{next_task_level_chorus}"
        )
        v2_batch_size = next_v2
        signal_encoder_batch_size = next_signal_encoder
        task_level_chorus_batch_size = next_task_level_chorus
        attempt += 1
