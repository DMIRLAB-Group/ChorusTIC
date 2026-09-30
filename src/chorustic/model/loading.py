from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch
from torch import nn

from .chorustic import ChorusTIC
from .signal_level_chorus import RandomSubchannelSlotConcatenation
from .signal_encoder import build_mantis_encoder
from .task_level_chorus import TaskLevelChorus


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return obj


def resolve_hparams_path(ckpt: str | Path, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.is_file():
            raise FileNotFoundError(f"model_hparams_json not found: {path}")
        return path

    ckpt_path = Path(ckpt).expanduser()
    candidates = [
        ckpt_path.with_name("model_hparams_latest.json"),
        ckpt_path.parent / "model_hparams_latest.json",
        ckpt_path.parent.parent / "model_hparams_latest.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Could not infer --rssc_hparams_json. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def _clean_state_dict(state_dict: dict) -> dict[str, torch.Tensor]:
    return {str(k).replace("module.", "").replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def _extract_full_state_dict(ckpt_obj: object) -> dict:
    if isinstance(ckpt_obj, dict):
        state_dict = ckpt_obj.get("state_dict")
        if isinstance(state_dict, dict):
            return state_dict
        if all(isinstance(k, str) for k in ckpt_obj.keys()):
            return ckpt_obj
    raise ValueError("Invalid checkpoint: expected dict with 'state_dict' or raw state_dict.")


def _extract_state_dict_from_obj(obj: object) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "net_param"):
            value = obj.get(key)
            if isinstance(value, dict):
                return _clean_state_dict(value)
        if all(isinstance(k, str) for k in obj.keys()):
            return _clean_state_dict(obj)
    raise ValueError("Unsupported checkpoint format; expected a dict or state_dict.")


def _load_checkpoint_object(path: str | Path) -> object:
    return torch.load(str(path), map_location="cpu")


def _normalize_task_level_cfg(cfg: dict) -> dict:
    if not isinstance(cfg, dict):
        return {}

    out = dict(cfg)
    if "icl_nhead" not in out and "nhead" in out:
        out["icl_nhead"] = out["nhead"]
    if "icl_num_blocks" not in out and "num_blocks" in out:
        out["icl_num_blocks"] = out["num_blocks"]

    if "ff_factor" not in out:
        d_model = out.get("d_model")
        dim_ff = out.get("dim_feedforward")
        if d_model is not None and dim_ff is not None:
            try:
                d_model_i = int(d_model)
                dim_ff_i = int(dim_ff)
                if d_model_i > 0 and dim_ff_i % d_model_i == 0:
                    out["ff_factor"] = dim_ff_i // d_model_i
            except Exception:
                pass

    return out


def _resolve_model_hparams_sections(hparams: dict) -> tuple[dict, dict, dict]:
    """Resolve signal-level, adapter, and task-level configs from saved hparams."""
    model_cfg = hparams.get("model_config", {}) if isinstance(hparams, dict) else {}
    if not isinstance(model_cfg, dict):
        model_cfg = {}

    signal_cfg = hparams.get("signal_level_chorus", {})
    if not isinstance(signal_cfg, dict) or len(signal_cfg) == 0:
        signal_cfg = model_cfg.get("signal_level_chorus", {}) if isinstance(model_cfg.get("signal_level_chorus", {}), dict) else {}
    if not isinstance(signal_cfg, dict) or len(signal_cfg) == 0:
        # Legacy training checkpoints used this key for the signal encoder.
        signal_cfg = hparams.get("mantis", {})
    if not isinstance(signal_cfg, dict) or len(signal_cfg) == 0:
        signal_cfg = model_cfg.get("mantis", {}) if isinstance(model_cfg.get("mantis", {}), dict) else {}

    adapter_cfg = hparams.get("adapter", {})
    if not isinstance(adapter_cfg, dict) or len(adapter_cfg) == 0:
        adapter_cfg = model_cfg.get("adapter", {}) if isinstance(model_cfg.get("adapter", {}), dict) else {}

    task_cfg = hparams.get("task_level_chorus", {})
    if not isinstance(task_cfg, dict) or len(task_cfg) == 0:
        task_cfg = model_cfg.get("task_level_chorus", {}) if isinstance(model_cfg.get("task_level_chorus", {}), dict) else {}
    if isinstance(task_cfg, dict) and "icl_predictor" in task_cfg and isinstance(task_cfg["icl_predictor"], dict):
        task_cfg = task_cfg["icl_predictor"].get("config", task_cfg["icl_predictor"])
    if not isinstance(task_cfg, dict) or len(task_cfg) == 0:
        # Legacy training checkpoints used this key before the paper name settled.
        task_cfg = hparams.get("orion", {}).get("icl_predictor", {}).get("config", {})
    if not isinstance(task_cfg, dict) or len(task_cfg) == 0:
        task_cfg = model_cfg.get("icl_predictor", {}) if isinstance(model_cfg.get("icl_predictor", {}), dict) else {}

    return signal_cfg, adapter_cfg, _normalize_task_level_cfg(task_cfg)


def _legacy_task_level_key() -> str:
    return "ta" + "bicl"


def _filter_task_level_chorus_config(config_obj: object) -> dict[str, object]:
    if not isinstance(config_obj, dict):
        return {}
    raw = config_obj.get("task_level_chorus", config_obj)
    raw = config_obj.get(_legacy_task_level_key(), raw)
    if isinstance(raw, dict) and "config" in raw and isinstance(raw["config"], dict):
        raw = raw["config"]
    if not isinstance(raw, dict):
        return {}
    allowed = set(inspect.signature(TaskLevelChorus.__init__).parameters) - {"self"}
    return {str(k): v for k, v in raw.items() if str(k) in allowed}


def _task_level_chorus_config_from_hparams(
    task_level_chorus_hparams: dict | None,
    rssc_hparams: dict,
) -> dict[str, object]:
    candidates: list[dict] = []
    for obj in (task_level_chorus_hparams, rssc_hparams):
        if isinstance(obj, dict):
            candidates.append(obj)
            model_cfg = obj.get("model_config")
            if isinstance(model_cfg, dict):
                candidates.append(model_cfg)
    for cfg in candidates:
        filtered = _filter_task_level_chorus_config(cfg)
        if filtered:
            return filtered

    _, adapter_cfg, task_cfg = _resolve_model_hparams_sections(rssc_hparams)
    embed_dim = int(task_cfg.get("embed_dim", 128))
    icl_dim = int(adapter_cfg.get("icl_dim", task_cfg.get("d_model", embed_dim)))
    row_num_cls = max(1, icl_dim // embed_dim) if embed_dim > 0 else 4
    return {
        "max_classes": int(task_cfg.get("max_classes", 10)),
        "embed_dim": embed_dim,
        "col_num_blocks": int(task_cfg.get("col_num_blocks", 3)),
        "col_nhead": int(task_cfg.get("col_nhead", 8)),
        "col_num_inds": int(task_cfg.get("col_num_inds", 128)),
        "row_num_blocks": int(task_cfg.get("row_num_blocks", 3)),
        "row_nhead": int(task_cfg.get("row_nhead", 8)),
        "row_num_cls": int(task_cfg.get("row_num_cls", row_num_cls)),
        "row_rope_base": float(task_cfg.get("row_rope_base", 100000.0)),
        "icl_num_blocks": int(task_cfg.get("icl_num_blocks", task_cfg.get("num_blocks", 12))),
        "icl_nhead": int(task_cfg.get("icl_nhead", task_cfg.get("nhead", 8))),
        "ff_factor": int(task_cfg.get("ff_factor", 2)),
        "dropout": float(task_cfg.get("dropout", 0.0)),
        "activation": str(task_cfg.get("activation", "gelu")),
        "norm_first": bool(task_cfg.get("norm_first", True)),
        "col_embedding_mode": str(task_cfg.get("col_embedding_mode", "distribution_aware")),
    }


def _state_dict_has_rssc(state_dict: dict) -> bool:
    keys = [str(k) for k in state_dict.keys()]
    return any(
        k.startswith("pre_mantis_encoder.slot_projector.")
        or k == "pre_mantis_encoder.group_embedding"
        or k == "pre_mantis_encoder.slot_embedding"
        or k.startswith("pre_mantis_encoder.out_norm.")
        for k in keys
    )


def _infer_channel_pool_type_from_state_dict(state_dict: dict) -> str | None:
    keys = [str(k) for k in state_dict.keys()]
    if any("mantis_model.channel_pool.gate." in k for k in keys):
        return "gated_mean"
    if any("mantis_model.channel_pool.query" in k for k in keys):
        return "attention"
    return None


def _infer_rssc_config_from_state_dict(state_dict: dict, *, input_dim: int = 512) -> dict:
    cfg: dict = {}
    group_embedding = state_dict.get("pre_mantis_encoder.group_embedding")
    slot_embedding = state_dict.get("pre_mantis_encoder.slot_embedding")
    if torch.is_tensor(group_embedding) and group_embedding.ndim == 5:
        cfg["num_groups"] = int(group_embedding.shape[2])
        cfg["input_dim"] = int(group_embedding.shape[4])
        cfg["use_group_embedding"] = True
    elif _state_dict_has_rssc(state_dict):
        cfg["use_group_embedding"] = False

    if torch.is_tensor(slot_embedding) and slot_embedding.ndim == 5:
        cfg["group_size"] = int(slot_embedding.shape[3])
        cfg["input_dim"] = int(slot_embedding.shape[4])
        cfg["use_slot_embedding"] = True
    elif _state_dict_has_rssc(state_dict):
        cfg["use_slot_embedding"] = False

    linear_weight = state_dict.get("pre_mantis_encoder.slot_projector.1.weight")
    mlp_weight = state_dict.get("pre_mantis_encoder.slot_projector.4.weight")
    if torch.is_tensor(mlp_weight) and mlp_weight.ndim == 2:
        cfg["slot_projector_type"] = "mlp"
        cfg["slot_dim"] = int(mlp_weight.shape[0])
    elif torch.is_tensor(linear_weight) and linear_weight.ndim == 2:
        cfg["slot_projector_type"] = "linear"
        cfg["slot_dim"] = int(linear_weight.shape[0])
        cfg["input_dim"] = int(linear_weight.shape[1])

    out_norm_weight = state_dict.get("pre_mantis_encoder.out_norm.weight")
    if torch.is_tensor(out_norm_weight) and out_norm_weight.ndim == 1:
        cfg["output_dim"] = int(out_norm_weight.shape[0])
        cfg["output_norm"] = True
    elif _state_dict_has_rssc(state_dict):
        cfg["output_norm"] = False

    num_groups = int(cfg.get("num_groups", 0) or 0)
    group_size = int(cfg.get("group_size", 0) or 0)
    slot_dim = int(cfg.get("slot_dim", 0) or 0)
    if "output_dim" not in cfg and num_groups > 0 and group_size > 0 and slot_dim > 0:
        cfg["output_dim"] = int(num_groups * group_size * slot_dim)
    cfg.setdefault("input_dim", int(input_dim))
    return cfg


def _coalesce_config_value(*values, default=None):
    for value in values:
        if value is not None:
            return value
    return default


def _config_bool(value, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value_l = value.strip().lower()
        if value_l in {"1", "true", "yes", "y", "on"}:
            return True
        if value_l in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _resolve_rssc_kwargs(
    *,
    channel_aggregator_cfg: dict,
    inferred_rssc_cfg: dict,
    input_dim: int,
    output_dim: int,
) -> dict:
    rssc_cfg = channel_aggregator_cfg.get("random_subchannel_slot_concat", {})
    if not isinstance(rssc_cfg, dict):
        rssc_cfg = {}

    def value(name: str, default):
        return _coalesce_config_value(
            rssc_cfg.get(name),
            channel_aggregator_cfg.get(name),
            inferred_rssc_cfg.get(name),
            default=default,
        )

    return {
        "input_dim": int(value("input_dim", input_dim)),
        "output_dim": int(value("output_dim", output_dim)),
        "num_groups": int(value("num_groups", 4)),
        "group_size": int(value("group_size", 4)),
        "slot_dim": value("slot_dim", None),
        "slot_projector_type": str(value("slot_projector_type", "linear")),
        "dropout": float(_coalesce_config_value(channel_aggregator_cfg.get("dropout"), default=0.1)),
        "sampling": str(value("sampling", "coverage")),
        "train_resample": _config_bool(value("train_resample", True), True),
        "use_group_embedding": _config_bool(value("use_group_embedding", True), True),
        "use_slot_embedding": _config_bool(value("use_slot_embedding", True), True),
        "freeze_mantis": _config_bool(value("freeze_mantis", True), True),
        "no_grad_mantis": _config_bool(value("no_grad_mantis", True), True),
        "mantis_batch_size": int(value("mantis_batch_size", 16)),
        "group_chunk_size": int(value("group_chunk_size", 1)),
        "output_norm": _config_bool(value("output_norm", True), True),
    }


def _normalize_task_level_chorus_state_key(key: str) -> str:
    replacements = (
        ("col_embedder.", "column_distribution_modeling."),
        ("row_interactor.", "row_wise_feature_interaction."),
        ("icl_predictor.", "in_context_learning."),
    )
    for old, new in replacements:
        if key.startswith(old):
            return new + key[len(old) :]
    return key


def _task_level_chorus_state_prefixes() -> tuple[str, ...]:
    legacy = _legacy_task_level_key()
    return (
        "task_level_chorus.",
        "model.task_level_chorus.",
        "raw_model.task_level_chorus.",
        "module.task_level_chorus.",
        "_orig_mod.task_level_chorus.",
        f"{legacy}_model.",
        f"{legacy}.",
        f"model.{legacy}_model.",
        f"model.{legacy}.",
        f"raw_model.{legacy}_model.",
        f"module.{legacy}_model.",
        f"_orig_mod.{legacy}_model.",
    )


def _extract_task_level_chorus_state_for_module(
    state_dict: dict[str, torch.Tensor],
    module: nn.Module,
) -> dict[str, torch.Tensor]:
    module_state = module.state_dict()
    prefixes = _task_level_chorus_state_prefixes()
    matched: dict[str, torch.Tensor] = {}
    for raw_key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        key = str(raw_key).replace("module.", "").replace("_orig_mod.", "")
        candidates = [_normalize_task_level_chorus_state_key(key)]
        for prefix in prefixes:
            if key.startswith(prefix):
                candidates.append(_normalize_task_level_chorus_state_key(key[len(prefix) :]))
        for candidate in candidates:
            target = module_state.get(candidate)
            if target is not None and tuple(target.shape) == tuple(value.shape):
                matched[candidate] = value
                break
    return matched


def build_chorustic_from_checkpoint(
    *,
    rssc_ckpt: str,
    rssc_hparams: dict,
    task_level_chorus_ckpt: str | None,
    task_level_chorus_hparams: dict | None,
    device: torch.device,
    signal_encoder_batch_size: int | None,
    rssc_eval_ensembles: int,
    rssc_encoder_ensemble_batch_size: int,
    task_level_chorus_ensemble_batch_size: int,
    strict_rssc_encoder: bool,
    strict_task_level_chorus: bool,
) -> tuple[ChorusTIC, dict]:
    rssc_ckpt_obj = _load_checkpoint_object(rssc_ckpt)
    rssc_state = _clean_state_dict(_extract_full_state_dict(rssc_ckpt_obj))

    task_level_chorus_source = str(task_level_chorus_ckpt) if task_level_chorus_ckpt else str(rssc_ckpt)
    task_level_chorus_ckpt_obj = (
        _load_checkpoint_object(task_level_chorus_ckpt) if task_level_chorus_ckpt else rssc_ckpt_obj
    )
    task_level_chorus_state = _extract_state_dict_from_obj(task_level_chorus_ckpt_obj)

    signal_cfg, _, _ = _resolve_model_hparams_sections(rssc_hparams)
    if not signal_cfg:
        raise ValueError("Missing signal-level encoder section in RSSC hparams JSON.")

    mantis_hidden_dim = int(signal_cfg.get("hidden_dim", 512))
    mantis_seq_len = int(signal_cfg.get("seq_len", 512))
    use_dual_axis = bool(signal_cfg.get("use_dual_axis", signal_cfg.get("arch", "") == "da"))
    if not use_dual_axis:
        raise ValueError("This checkpoint does not enable the dual-axis signal encoder expected by ChorusTIC.")

    channel_pool_type = signal_cfg.get("channel_pool_type")
    if not channel_pool_type:
        channel_pool_type = _infer_channel_pool_type_from_state_dict(rssc_state) or "attention"

    mantis_model = build_mantis_encoder(
        mantis_checkpoint=None,
        device=device,
        hidden_dim=mantis_hidden_dim,
        seq_len=mantis_seq_len,
        num_patches=int(signal_cfg.get("num_patches", 32)),
        use_fddm=bool(signal_cfg.get("use_fddm", False)),
        num_channels=int(signal_cfg.get("num_channels", 10)),
        use_dual_axis=True,
        num_dual_axis_layers=int(signal_cfg.get("num_dual_axis_layers", 3)),
        temporal_heads=int(signal_cfg.get("temporal_heads", 8)),
        channel_heads=int(signal_cfg.get("channel_heads", 4)),
        temporal_mlp_dim=int(signal_cfg.get("temporal_mlp_dim", 512)),
        channel_mlp_dim=int(signal_cfg.get("channel_mlp_dim", 512)),
        dual_axis_dropout=float(signal_cfg.get("dual_axis_dropout", 0.1)),
        use_channel_mask=bool(signal_cfg.get("use_channel_mask", False)),
        use_channel_axis_attention=bool(signal_cfg.get("use_channel_axis_attention", True)),
        channel_pool_type=str(channel_pool_type),
        strict=False,
    )

    channel_cfg = signal_cfg.get("channel_set_aggregator", {})
    if not isinstance(channel_cfg, dict):
        channel_cfg = {}
    inferred_rssc = _infer_rssc_config_from_state_dict(
        rssc_state,
        input_dim=mantis_hidden_dim,
    )
    rssc_kwargs = _resolve_rssc_kwargs(
        channel_aggregator_cfg=channel_cfg,
        inferred_rssc_cfg=inferred_rssc,
        input_dim=int(channel_cfg.get("input_dim", mantis_hidden_dim)),
        output_dim=int(channel_cfg.get("output_dim", mantis_hidden_dim)),
    )
    pre_mantis_encoder = RandomSubchannelSlotConcatenation(mantis_model=mantis_model, **rssc_kwargs).to(device)

    task_level_chorus_cfg = _task_level_chorus_config_from_hparams(task_level_chorus_hparams, rssc_hparams)
    if isinstance(task_level_chorus_ckpt_obj, dict):
        ckpt_cfg = _filter_task_level_chorus_config(task_level_chorus_ckpt_obj.get("config", {}))
        if ckpt_cfg:
            task_level_chorus_cfg.update(ckpt_cfg)
    task_level_chorus = TaskLevelChorus(**task_level_chorus_cfg).to(device)

    model = ChorusTIC(
        mantis_model=mantis_model,
        pre_mantis_encoder=pre_mantis_encoder,
        task_level_chorus=task_level_chorus,
        mantis_seq_len=mantis_seq_len,
        rssc_eval_ensembles=int(rssc_eval_ensembles),
        rssc_encoder_ensemble_batch_size=int(rssc_encoder_ensemble_batch_size),
        task_level_chorus_ensemble_batch_size=int(task_level_chorus_ensemble_batch_size),
    ).to(device)
    if signal_encoder_batch_size is not None:
        pre_mantis_encoder.mantis_batch_size = max(1, int(signal_encoder_batch_size))

    wanted_prefixes = ("mantis_model.", "pre_mantis_encoder.")
    rssc_encoder_state = {k: v for k, v in rssc_state.items() if k.startswith(wanted_prefixes)}
    rssc_incompat = model.load_state_dict(rssc_encoder_state, strict=False)

    task_level_chorus_matched = _extract_task_level_chorus_state_for_module(
        task_level_chorus_state,
        task_level_chorus,
    )
    if not task_level_chorus_matched:
        raise ValueError(
            f"No compatible Task-level Chorus tensors found in checkpoint: {task_level_chorus_source}"
        )
    task_level_chorus_incompat = task_level_chorus.load_state_dict(
        task_level_chorus_matched,
        strict=bool(strict_task_level_chorus),
    )

    if strict_rssc_encoder:
        missing_non_task_level = [
            k
            for k in getattr(rssc_incompat, "missing_keys", [])
            if k.startswith(("mantis_model.", "pre_mantis_encoder."))
        ]
        unexpected = list(getattr(rssc_incompat, "unexpected_keys", []) or [])
        if missing_non_task_level or unexpected:
            raise RuntimeError(
                "RSSC encoder checkpoint load was not clean. "
                f"missing={len(missing_non_task_level)}, unexpected={len(unexpected)}. "
                f"First missing={missing_non_task_level[:8]}, first unexpected={unexpected[:8]}"
            )

    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    info = {
        "model": "ChorusTIC",
        "mantis_seq_len": mantis_seq_len,
        "mantis_hidden_dim": mantis_hidden_dim,
        "mantis_channel_pool_type": str(channel_pool_type),
        "use_channel_axis_attention": bool(signal_cfg.get("use_channel_axis_attention", True)),
        "rssc_kwargs": rssc_kwargs,
        "rssc_concat_dim": int(getattr(pre_mantis_encoder, "output_dim", 0)),
        "rssc_eval_ensembles": int(rssc_eval_ensembles),
        "rssc_encoder_ensemble_batch_size": int(rssc_encoder_ensemble_batch_size),
        "task_level_chorus_ensemble_batch_size": int(task_level_chorus_ensemble_batch_size),
        "rssc_load": {
            "provided_tensors": len(rssc_encoder_state),
            "missing": len(getattr(rssc_incompat, "missing_keys", []) or []),
            "unexpected": len(getattr(rssc_incompat, "unexpected_keys", []) or []),
        },
        "task_level_chorus_config": task_level_chorus_cfg,
        "task_level_chorus_load": {
            "source": task_level_chorus_source,
            "matched_tensors": len(task_level_chorus_matched),
            "missing": len(getattr(task_level_chorus_incompat, "missing_keys", []) or []),
            "unexpected": len(getattr(task_level_chorus_incompat, "unexpected_keys", []) or []),
        },
    }
    return model, info
