from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from .task_level_chorus import InferenceConfig, TaskLevelChorus

from .signal_level_chorus import RandomSubchannelSlotConcatenation


class ChorusTIC(nn.Module):
    """ChorusTIC inference model.

    The module follows the paper's two-level decomposition:
    signal-level Chorus maps variable-channel time series into fixed-width RSSC
    representations, then task-level Chorus performs leakage-protected
    in-context classification through CDM, row-wise feature interaction, and ICL.
    """

    def __init__(
        self,
        *,
        mantis_model: nn.Module,
        pre_mantis_encoder: RandomSubchannelSlotConcatenation,
        task_level_chorus: TaskLevelChorus,
        mantis_seq_len: int = 512,
        rssc_eval_ensembles: int = 1,
        rssc_encoder_ensemble_batch_size: int = 1,
        task_level_chorus_ensemble_batch_size: int = 64,
    ) -> None:
        super().__init__()
        self.mantis_model = mantis_model
        self.pre_mantis_encoder = pre_mantis_encoder
        self.task_level_chorus = task_level_chorus
        self.mantis_seq_len = int(mantis_seq_len)
        self.rssc_eval_ensembles = max(1, int(rssc_eval_ensembles))
        self.rssc_encoder_ensemble_batch_size = max(1, int(rssc_encoder_ensemble_batch_size))
        self.task_level_chorus_ensemble_batch_size = max(1, int(task_level_chorus_ensemble_batch_size))
        self.max_classes = int(getattr(task_level_chorus, "max_classes", 0) or 0)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and bool(getattr(self.pre_mantis_encoder, "freeze_mantis", False)):
            self.mantis_model.eval()
        return self

    @staticmethod
    def _normalize_channel_mask(
        X: Tensor,
        channel_mask: Optional[Tensor] = None,
        d: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        if X.ndim != 4:
            return None
        B, T, C, _ = X.shape
        if channel_mask is not None:
            if channel_mask.ndim != 3 or tuple(channel_mask.shape) != (B, T, C):
                raise ValueError(f"Expected channel_mask with shape {(B, T, C)}, got {tuple(channel_mask.shape)}")
            return channel_mask.to(device=X.device, dtype=torch.bool)
        if d is None:
            return None
        if d.ndim == 3 and tuple(d.shape) == (B, T, C):
            return d.to(device=X.device, dtype=torch.bool)
        idx = torch.arange(C, device=X.device).view(1, 1, C)
        if d.ndim == 2 and tuple(d.shape) == (B, T):
            counts = d.to(device=X.device, dtype=torch.long).clamp(min=0, max=C)
            return idx < counts.unsqueeze(-1)
        if d.ndim == 1 and int(d.shape[0]) == B:
            counts = d.to(device=X.device, dtype=torch.long).clamp(min=0, max=C)
            return (idx < counts.view(B, 1, 1)).expand(B, T, C)
        return None

    def _resize_last_dim(self, X: Tensor) -> Tensor:
        target = int(self.mantis_seq_len)
        if X.shape[-1] == target:
            return X
        original_shape = X.shape
        original_dtype = X.dtype
        X_reshaped = X.reshape(-1, 1, original_shape[-1])
        if not X_reshaped.is_floating_point():
            X_reshaped = X_reshaped.float()
        X_resized = torch.nn.functional.interpolate(
            X_reshaped,
            size=target,
            mode="linear",
            align_corners=False,
        )
        X_resized = X_resized.reshape(*original_shape[:-1], target)
        if X_resized.dtype != original_dtype:
            X_resized = X_resized.to(original_dtype)
        return X_resized

    def _prepare_signal_level_input(
        self,
        X: Tensor,
        *,
        d: Optional[Tensor] = None,
        channel_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        if X.ndim == 3:
            X = X.unsqueeze(2)
            normalized_mask = None
        elif X.ndim == 4:
            normalized_mask = self._normalize_channel_mask(X, channel_mask=channel_mask, d=d)
        else:
            raise ValueError(f"Expected X to be (B,T,L) or (B,T,C,L), got {tuple(X.shape)}")
        return self._resize_last_dim(X), normalized_mask

    def _encode_signal_level_chorus(self, X: Tensor, channel_mask: Optional[Tensor]) -> Tensor:
        reps = self.pre_mantis_encoder(
            X,
            train_size=int(X.shape[1]),
            channel_mask=channel_mask,
        )
        return reps.reshape(X.shape[0], X.shape[1], -1).to(X.device)

    @staticmethod
    def _default_task_level_inference_config(device: torch.device) -> InferenceConfig:
        config = InferenceConfig()
        config.update_from_dict(
            {
                "COL_CONFIG": {"device": device},
                "ROW_CONFIG": {"device": device},
                "ICL_CONFIG": {"device": device},
            }
        )
        return config

    def encode_signal_level_chorus(
        self,
        X: Tensor,
        *,
        d: Optional[Tensor] = None,
        channel_mask: Optional[Tensor] = None,
    ) -> Tensor:
        X, normalized_mask = self._prepare_signal_level_input(X, d=d, channel_mask=channel_mask)
        return self._encode_signal_level_chorus(X, normalized_mask)

    def forward(
        self,
        X: Tensor,
        y_train: Tensor,
        d: Optional[Tensor] = None,
        channel_mask: Optional[Tensor] = None,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        inference_config=None,
        **_unused,
    ) -> Tensor:
        n_ensembles = int(self.rssc_eval_ensembles) if not self.training else 1
        n_ensembles = max(1, n_ensembles)
        X, normalized_mask = self._prepare_signal_level_input(X, d=d, channel_mask=channel_mask)
        if inference_config is None and not self.training:
            inference_config = self._default_task_level_inference_config(X.device)
        if n_ensembles == 1:
            task_level_features = self._encode_signal_level_chorus(X, normalized_mask)
            return self.task_level_chorus(
                task_level_features,
                y_train=y_train,
                d=None,
                return_logits=return_logits,
                softmax_temperature=softmax_temperature,
                inference_config=inference_config,
            )

        batch_size = int(X.shape[0])
        encoder_chunk_size = min(int(self.rssc_encoder_ensemble_batch_size), n_ensembles)
        task_level_feature_batches = []
        for start in range(0, n_ensembles, encoder_chunk_size):
            count = min(encoder_chunk_size, n_ensembles - start)
            if count == 1:
                task_level_feature_batches.append(self._encode_signal_level_chorus(X, normalized_mask))
                continue

            X_repeated = X.repeat((count,) + (1,) * (X.ndim - 1))
            mask_repeated = None
            if normalized_mask is not None:
                mask_repeated = normalized_mask.repeat((count,) + (1,) * (normalized_mask.ndim - 1))
            task_level_feature_batches.append(self._encode_signal_level_chorus(X_repeated, mask_repeated))

        task_level_features = torch.cat(task_level_feature_batches, dim=0)
        y_train_repeated = y_train.repeat((n_ensembles,) + (1,) * (y_train.ndim - 1))

        outs = []
        chunk_size = min(int(self.task_level_chorus_ensemble_batch_size), int(task_level_features.shape[0]))
        for start in range(0, int(task_level_features.shape[0]), chunk_size):
            end = start + chunk_size
            outs.append(
                self.task_level_chorus(
                    task_level_features[start:end],
                    y_train=y_train_repeated[start:end],
                    d=None,
                    return_logits=return_logits,
                    softmax_temperature=softmax_temperature,
                    inference_config=inference_config,
                )
            )
        out = torch.cat(outs, dim=0)
        return out.reshape(n_ensembles, batch_size, *out.shape[1:]).mean(dim=0)
