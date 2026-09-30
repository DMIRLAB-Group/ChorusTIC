from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def _extract_state_dict(ckpt_obj: Any) -> dict[str, torch.Tensor]:
    """Best-effort extraction of a PyTorch state_dict from a checkpoint object."""
    if isinstance(ckpt_obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            val = ckpt_obj.get(key)
            if isinstance(val, dict):
                return {str(k).replace("module.", ""): v for k, v in val.items()}
        # Some checkpoints are already a state-dict mapping.
        if all(isinstance(k, str) for k in ckpt_obj.keys()):
            return {str(k).replace("module.", ""): v for k, v in ckpt_obj.items()}
    raise ValueError("Unsupported mantis checkpoint format; expected a dict or state_dict.")


def build_mantis_encoder(
    *,
    mantis_checkpoint: str | Path | None,
    device: torch.device | str | None = None,
    hidden_dim: int = 512,
    seq_len: int = 512,
    num_patches: int = 32,
    use_fddm: bool = False,
    num_channels: int = 1,
    use_dual_axis: bool = False,
    num_dual_axis_layers: int = 3,
    temporal_heads: int = 8,
    channel_heads: int = 4,
    temporal_mlp_dim: int = 512,
    channel_mlp_dim: int = 512,
    dual_axis_dropout: float = 0.1,
    use_channel_mask: bool = False,
    use_channel_axis_attention: bool = True,
    channel_pool_type: str | None = None,
    strict: bool = False,
) -> nn.Module:
    """Build a Mantis encoder and optionally load a checkpoint.

    This repo uses the shared implementation under `chorustic.model.signal_encoder.TSEncoder`.
    The returned module accepts input shaped `(B, C, L)` where `L == seq_len`.
    """

    dev = torch.device(device) if device is not None else torch.device("cpu")
    ckpt_path = Path(mantis_checkpoint) if mantis_checkpoint is not None else None

    resolved_channel_pool_type = channel_pool_type
    if use_dual_axis and resolved_channel_pool_type is None and ckpt_path is not None and ckpt_path.is_file():
        ckpt_obj = torch.load(str(ckpt_path), map_location="cpu")
        state_dict = _extract_state_dict(ckpt_obj)
        if any(str(k).endswith("channel_pool.gate.0.weight") or "channel_pool.gate." in str(k) for k in state_dict):
            resolved_channel_pool_type = "gated_mean"
        else:
            resolved_channel_pool_type = "attention"
    elif resolved_channel_pool_type is None:
        resolved_channel_pool_type = "attention"

    # Import lazily: TSEncoder pulls in heavier deps (einops, huggingface_hub, etc.).
    from chorustic.model.signal_encoder.TSEncoder.architecture.architecture import Mantis8M, Mantis8MWithFDDM, MantisDA

    if use_dual_axis:
        model = MantisDA(
            seq_len=int(seq_len),
            hidden_dim=int(hidden_dim),
            num_patches=int(num_patches),
            num_input_channels=int(num_channels),
            num_dual_axis_layers=int(num_dual_axis_layers),
            temporal_heads=int(temporal_heads),
            channel_heads=int(channel_heads),
            temporal_mlp_dim=int(temporal_mlp_dim),
            channel_mlp_dim=int(channel_mlp_dim),
            dropout=float(dual_axis_dropout),
            use_channel_mask=bool(use_channel_mask),
            use_channel_axis_attention=bool(use_channel_axis_attention),
            channel_pool_type=str(resolved_channel_pool_type),
            device=str(dev),
            pre_training=False,
        )
    elif use_fddm:
        model: nn.Module = Mantis8MWithFDDM(
            seq_len=int(seq_len),
            hidden_dim=int(hidden_dim),
            num_patches=int(num_patches),
            num_channels=int(num_channels),
            device=str(dev),
            pre_training=False,
        )
    else:
        model = Mantis8M(
            seq_len=int(seq_len),
            hidden_dim=int(hidden_dim),
            num_patches=int(num_patches),
            device=str(dev),
            pre_training=False,
        )

    if ckpt_path is not None:
        if ckpt_path.is_dir():
            try:
                model = model.from_pretrained(str(ckpt_path))
            except Exception:
                if not use_dual_axis:
                    raise
                base_model = Mantis8M(
                    seq_len=int(seq_len),
                    hidden_dim=int(hidden_dim),
                    num_patches=int(num_patches),
                    device=str(dev),
                    pre_training=False,
                ).from_pretrained(str(ckpt_path))
                model.load_from_mantis(base_model)
            print(f"[SignalEncoder] Loaded pretrained Mantis encoder from {ckpt_path}")
        elif ckpt_path.is_file():
            ckpt_obj = torch.load(str(ckpt_path), map_location="cpu")
            state_dict = _extract_state_dict(ckpt_obj)
            model.load_state_dict(state_dict, strict=bool(strict))
        else:
            raise FileNotFoundError(f"Mantis checkpoint not found: {ckpt_path}")

    model.to(dev)
    model.eval()
    return model


@torch.no_grad()
def encode_with_mantis(
    model: nn.Module,
    x: torch.Tensor,
    *,
    batch_size: int = 256,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Encode a batch of time series with a Mantis encoder.

    - `x` supports `(N, L)`, `(N, 1, L)` or `(N, C, L)`.
    - returns `(N, D)` where `D == model.hidden_dim`.
    """

    if x.dim() == 2:
        x = x[:, None, :]
    if x.dim() != 3:
        raise ValueError(f"Expected x of shape (N,L) or (N,C,L); got {tuple(x.shape)}")

    dev = torch.device(device) if device is not None else next(model.parameters()).device
    x = x.to(dev)

    outs: list[torch.Tensor] = []
    bs = max(1, int(batch_size))
    for i in range(0, x.shape[0], bs):
        outs.append(model(x[i : i + bs]))
    return torch.cat(outs, dim=0)
