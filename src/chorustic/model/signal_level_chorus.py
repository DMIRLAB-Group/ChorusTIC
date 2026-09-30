from __future__ import annotations

import contextlib
import math
from typing import Optional

import torch
from torch import Tensor, nn


class RandomSubchannelSlotConcatenation(nn.Module):
    """Signal-level Chorus RSSC composition.

    Input is ``x`` with shape ``(B, T, C, L)``. For each episode ``B``, RSSC
    samples ``num_groups`` channel groups with ``group_size`` slots, feeds each
    group through the shared dual-axis encoder, projects each slot to
    ``slot_dim``, and concatenates all slots in a fixed order.
    """

    def __init__(
        self,
        *,
        mantis_model: nn.Module,
        input_dim: int = 512,
        output_dim: int = 512,
        num_groups: int = 4,
        group_size: int = 4,
        slot_dim: Optional[int] = None,
        slot_projector_type: str = "linear",
        dropout: float = 0.1,
        sampling: str = "coverage",
        train_resample: bool = True,
        use_group_embedding: bool = True,
        use_slot_embedding: bool = True,
        freeze_mantis: bool = True,
        no_grad_mantis: bool = True,
        mantis_batch_size: int = 16,
        group_chunk_size: int = 1,
        output_norm: bool = True,
    ) -> None:
        super().__init__()
        if not hasattr(mantis_model, "get_channel_features"):
            raise ValueError("RSSC requires a dual-axis encoder with get_channel_features().")

        input_dim = int(input_dim)
        output_dim = int(output_dim)
        num_groups = int(num_groups)
        group_size = int(group_size)
        if input_dim <= 0:
            raise ValueError(f"input_dim must be > 0, got {input_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be > 0, got {output_dim}")
        if num_groups <= 0:
            raise ValueError(f"num_groups must be > 0, got {num_groups}")
        if group_size <= 0:
            raise ValueError(f"group_size must be > 0, got {group_size}")

        num_slots = num_groups * group_size
        if slot_dim is None:
            if output_dim % num_slots != 0:
                raise ValueError(
                    f"output_dim={output_dim} must be divisible by "
                    f"num_groups*group_size={num_slots}"
                )
            slot_dim = output_dim // num_slots
        else:
            slot_dim = int(slot_dim)
            if num_slots * slot_dim != output_dim:
                raise ValueError(
                    "num_groups*group_size*slot_dim must equal output_dim, "
                    f"got {num_groups}*{group_size}*{slot_dim} != {output_dim}"
                )
        if slot_dim <= 0:
            raise ValueError(f"slot_dim must be > 0, got {slot_dim}")

        slot_projector_type = str(slot_projector_type).lower()
        if slot_projector_type == "linear":
            self.slot_projector = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, slot_dim),
            )
        elif slot_projector_type == "mlp":
            hidden = min(256, input_dim)
            self.slot_projector = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden, slot_dim),
            )
        else:
            raise ValueError(f"slot_projector_type must be 'linear' or 'mlp', got {slot_projector_type!r}")

        sampling = str(sampling).lower()
        if sampling not in {"coverage", "random"}:
            raise ValueError(f"sampling must be 'coverage' or 'random', got {sampling!r}")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_groups = num_groups
        self.group_size = group_size
        self.num_slots = num_slots
        self.slot_dim = int(slot_dim)
        self.slot_projector_type = slot_projector_type
        self.dropout = float(dropout)
        self.sampling = sampling
        self.train_resample = bool(train_resample)
        self.use_group_embedding = bool(use_group_embedding)
        self.use_slot_embedding = bool(use_slot_embedding)
        self.freeze_mantis = bool(freeze_mantis)
        self.no_grad_mantis = bool(no_grad_mantis)
        self.mantis_batch_size = max(1, int(mantis_batch_size))
        self.group_chunk_size = max(1, int(group_chunk_size))
        self.output_norm_enabled = bool(output_norm)

        # Keep checkpoint keys as `mantis_model.*` on the ChorusTIC wrapper, not
        # duplicated under `pre_mantis_encoder.mantis_model.*`.
        object.__setattr__(self, "mantis_model", mantis_model)

        if self.use_group_embedding:
            self.group_embedding = nn.Parameter(torch.randn(1, 1, num_groups, 1, input_dim) * 0.02)
        if self.use_slot_embedding:
            self.slot_embedding = nn.Parameter(torch.randn(1, 1, 1, group_size, input_dim) * 0.02)

        self.out_norm = nn.LayerNorm(output_dim) if self.output_norm_enabled else nn.Identity()
        self._cached_group_idx: Tensor | None = None
        self.last_group_idx: Tensor | None = None

        if self.freeze_mantis:
            for p in self.mantis_model.parameters():
                p.requires_grad_(False)
            self.mantis_model.eval()

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"num_groups={self.num_groups}, group_size={self.group_size}, slot_dim={self.slot_dim}, "
            f"sampling={self.sampling}, mantis_batch_size={self.mantis_batch_size}, "
            f"group_chunk_size={self.group_chunk_size}, freeze_mantis={self.freeze_mantis}, "
            f"no_grad_mantis={self.no_grad_mantis}"
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_mantis:
            self.mantis_model.eval()
        return self

    def _mantis_supports_channel_mask(self) -> bool:
        return bool(getattr(self.mantis_model, "use_channel_mask", False))

    def _sample_group_indices_uncached(
        self,
        channel_mask: torch.Tensor | None,
        B: int,
        T: int,
        C: int,
        device: torch.device,
    ) -> torch.Tensor:
        if channel_mask is None:
            valid_channel = torch.ones(B, C, dtype=torch.bool, device=device)
        else:
            if channel_mask.shape != (B, T, C):
                raise ValueError(f"Expected channel_mask with shape {(B, T, C)}, got {tuple(channel_mask.shape)}")
            valid_channel = channel_mask.to(device=device, dtype=torch.bool).any(dim=1)

        num_need = self.num_groups * self.group_size
        group_idx = torch.empty(B, self.num_groups, self.group_size, dtype=torch.long, device=device)

        for b in range(B):
            valid_idx = torch.where(valid_channel[b])[0]
            if valid_idx.numel() == 0:
                valid_idx = torch.tensor([0], device=device, dtype=torch.long)

            if valid_idx.numel() == 1:
                idx = valid_idx.repeat(num_need)
            elif valid_idx.numel() < num_need:
                if self.sampling == "coverage":
                    pieces = []
                    repeat_times = int(math.ceil(num_need / int(valid_idx.numel())))
                    for _ in range(repeat_times):
                        pieces.append(valid_idx[torch.randperm(int(valid_idx.numel()), device=device)])
                    idx = torch.cat(pieces, dim=0)[:num_need]
                else:
                    rand = torch.randint(0, int(valid_idx.numel()), (num_need,), device=device)
                    idx = valid_idx[rand]
            else:
                if self.sampling == "coverage":
                    perm = valid_idx[torch.randperm(int(valid_idx.numel()), device=device)]
                    idx = perm[:num_need]
                else:
                    rand = torch.randint(0, int(valid_idx.numel()), (num_need,), device=device)
                    idx = valid_idx[rand]

            group_idx[b] = idx.reshape(self.num_groups, self.group_size)

        return group_idx

    def _sample_group_indices(
        self,
        channel_mask: torch.Tensor | None,
        B: int,
        T: int,
        C: int,
        device: torch.device,
    ) -> torch.Tensor:
        if C <= 0:
            raise ValueError(f"C must be > 0, got {C}")

        if self.training and not self.train_resample:
            cached = self._cached_group_idx
            if cached is not None and tuple(cached.shape) == (B, self.num_groups, self.group_size) and cached.device == device:
                cached_in_bounds = cached.numel() == 0 or int(cached.max().item()) < int(C)
                if cached_in_bounds and channel_mask is None:
                    return cached
                if cached_in_bounds and channel_mask is not None:
                    valid_channel = channel_mask.to(device=device, dtype=torch.bool).any(dim=1)
                    flat = cached.reshape(B, -1)
                    valid_for_cached = torch.gather(valid_channel, dim=1, index=flat).all()
                    if bool(valid_for_cached.item()):
                        return cached

        group_idx = self._sample_group_indices_uncached(channel_mask, B, T, C, device)
        if self.training and not self.train_resample:
            self._cached_group_idx = group_idx.detach()
        return group_idx

    def _gather_group_chunk(
        self,
        x: torch.Tensor,
        channel_mask: torch.Tensor | None,
        group_idx_chunk: torch.Tensor,
    ) -> tuple[Tensor, Tensor]:
        B, T, C, L = x.shape
        if group_idx_chunk.ndim != 3 or group_idx_chunk.shape[0] != B or group_idx_chunk.shape[2] != self.group_size:
            raise ValueError(
                "group_idx_chunk must have shape (B,Kc,group_size), got "
                f"{tuple(group_idx_chunk.shape)} for B={B}, group_size={self.group_size}"
            )
        Kc = int(group_idx_chunk.shape[1])

        idx = group_idx_chunk[:, None, :, :, None].expand(B, T, Kc, self.group_size, L)
        x_expand = x[:, :, None, :, :].expand(B, T, Kc, C, L)
        x_chunk = torch.gather(x_expand, dim=3, index=idx)

        if channel_mask is None:
            mask_chunk = torch.ones(B, T, Kc, self.group_size, dtype=torch.bool, device=x.device)
        else:
            if channel_mask.shape != (B, T, C):
                raise ValueError(f"Expected channel_mask with shape {(B, T, C)}, got {tuple(channel_mask.shape)}")
            channel_mask_bool = channel_mask.to(device=x.device, dtype=torch.bool)
            mask_expand = channel_mask_bool[:, :, None, :].expand(B, T, Kc, C)
            mask_idx = group_idx_chunk[:, None, :, :].expand(B, T, Kc, self.group_size)
            mask_chunk = torch.gather(mask_expand, dim=3, index=mask_idx)

            all_invalid = ~channel_mask_bool.any(dim=2)
            if bool(all_invalid.any().item()):
                mask_chunk = torch.where(all_invalid[:, :, None, None], torch.ones_like(mask_chunk), mask_chunk)

        return x_chunk, mask_chunk

    def forward(
        self,
        x: torch.Tensor,
        *,
        train_size: int,
        channel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x to be (B,T,C,L), got {tuple(x.shape)}")
        B, T, C, L = x.shape
        if C <= 0:
            raise ValueError(f"Input must have at least one channel, got C={C}")
        _ = int(train_size)

        group_idx = self._sample_group_indices(channel_mask, B, T, C, x.device)
        self.last_group_idx = group_idx.detach()

        small_chunks: list[Tensor] = []
        mantis_mask_enabled = self._mantis_supports_channel_mask()
        batch_size = max(1, int(self.mantis_batch_size))
        group_chunk_size = max(1, int(self.group_chunk_size))

        for g_start in range(0, self.num_groups, group_chunk_size):
            g_end = min(self.num_groups, g_start + group_chunk_size)
            Kc = g_end - g_start
            group_idx_chunk = group_idx[:, g_start:g_end, :]
            x_chunk, mask_chunk = self._gather_group_chunk(x, channel_mask, group_idx_chunk)

            x_flat = x_chunk.reshape(B * T * Kc, self.group_size, L)
            mask_flat = mask_chunk.reshape(B * T * Kc, self.group_size)

            ch_list: list[Tensor] = []
            for s in range(0, x_flat.shape[0], batch_size):
                xb = x_flat[s : s + batch_size]
                mb = mask_flat[s : s + batch_size]
                ctx = torch.no_grad() if self.no_grad_mantis else contextlib.nullcontext()
                with ctx:
                    ch = self.mantis_model.get_channel_features(
                        xb,
                        channel_mask=mb if mantis_mask_enabled else None,
                    )
                if ch.ndim != 3 or ch.shape[1] != self.group_size or ch.shape[2] != self.input_dim:
                    raise RuntimeError(
                        "dual-axis encoder returned unexpected shape "
                        f"{tuple(ch.shape)}; expected (batch,{self.group_size},{self.input_dim})"
                    )
                ch_list.append(ch)

            ch_flat = torch.cat(ch_list, dim=0)
            ch_full = ch_flat.reshape(B, T, Kc, self.group_size, self.input_dim)

            if self.use_group_embedding:
                ch_full = ch_full + self.group_embedding[:, :, g_start:g_end, :, :].to(dtype=ch_full.dtype)
            if self.use_slot_embedding:
                ch_full = ch_full + self.slot_embedding.to(dtype=ch_full.dtype)

            small_chunks.append(self.slot_projector(ch_full))

        z_small = torch.cat(small_chunks, dim=2)
        out = z_small.reshape(B, T, self.output_dim)
        return self.out_norm(out)
