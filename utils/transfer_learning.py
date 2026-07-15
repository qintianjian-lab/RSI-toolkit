import os
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn


def _get_base_module(model: nn.Module) -> nn.Module:
    """
    If model is torch.compile()'d, it is usually an OptimizedModule with `_orig_mod`.
    Operate on the base module so parameter names and submodules (e.g. `head`) are accessible.
    """
    return getattr(model, "_orig_mod", model)


def _strip_prefix(s: str, prefix: str) -> str:
    return s[len(prefix):] if s.startswith(prefix) else s


def _normalize_ckpt_key(k: str) -> str:
    # Lightning checkpoints usually prefix with "model."
    k = _strip_prefix(k, "model.")
    # torch.compile checkpoints may include "_orig_mod." in keys
    k = _strip_prefix(k, "_orig_mod.")
    return k


def _possible_target_keys(norm_ckpt_key: str) -> Tuple[str, ...]:
    """
    Given a normalized ckpt key (after stripping Lightning/compile prefixes),
    generate possible target keys in the current model's state_dict.

    This exists because pretraining scripts may save weights under:
      - encoder.stem..., encoder.stage1...   (SBMAutoEncoder / SBMEncoder)
      - ae.encoder.stem...                   (pretraining wrappers)
    while the downstream classifier expects:
      - stem..., stage1..., attnpool..., head...

    The reverse direction is also useful for some checkpoint inspection workflows.
    """
    k = norm_ckpt_key

    # Common "strip" variants (wrapper pretrain -> backbone classifier)
    strip_variants = []
    if k.startswith("encoder."):
        strip_variants.append(k[len("encoder."):])
    if k.startswith("encoder.backbone."):
        strip_variants.append(k[len("encoder.backbone."):])
    if k.startswith("ae.encoder."):
        strip_variants.append(k[len("ae.encoder."):])
    if k.startswith("ae.encoder.backbone."):
        strip_variants.append(k[len("ae.encoder.backbone."):])
    if k.startswith("model.encoder."):
        strip_variants.append(k[len("model.encoder."):])
    if k.startswith("model.encoder.backbone."):
        strip_variants.append(k[len("model.encoder.backbone."):])
    if k.startswith("model.ae.encoder."):
        strip_variants.append(k[len("model.ae.encoder."):])
    if k.startswith("model.ae.encoder.backbone."):
        strip_variants.append(k[len("model.ae.encoder.backbone."):])

    # Common "add" variants (backbone classifier -> wrapper pretrain)
    add_variants = [
        "encoder." + k,
        "encoder.backbone." + k,
        "ae.encoder." + k,
        "ae.encoder.backbone." + k,
    ]

    # Internal rename compatibility:
    # Older checkpoints may use `BandProcessor.eca`; newer variants may use `BandProcessor.attn`.
    # This changes state_dict key segments like ".eca." <-> ".attn." while keeping tensor shapes identical.
    rename_variants = []
    for kk in [k, *strip_variants]:
        if ".eca." in kk:
            rename_variants.append(kk.replace(".eca.", ".attn."))
        if ".attn." in kk:
            rename_variants.append(kk.replace(".attn.", ".eca."))

    # Prefer direct match, then stripped, then internal-renamed, then added (and their internal renames).
    out = [k]
    out.extend(strip_variants)
    out.extend(rename_variants)
    out.extend(add_variants)
    for kk in add_variants:
        if ".eca." in kk:
            out.append(kk.replace(".eca.", ".attn."))
        if ".attn." in kk:
            out.append(kk.replace(".attn.", ".eca."))

    # de-dup while preserving order
    seen = set()
    uniq = []
    for kk in out:
        if kk not in seen:
            seen.add(kk)
            uniq.append(kk)
    return tuple(uniq)


def load_pretrained_weights(
    model: nn.Module,
    checkpoint_path: str,
    strict: bool = False,
    shape_filter: bool = True
) -> Tuple[int, int]:
    """
    Load pretrained weights into `model` (or its _orig_mod), robust to Lightning and torch.compile prefixes.
    Returns (num_loaded_keys, num_total_keys_in_ckpt_after_norm).
    """
    if not checkpoint_path or not isinstance(checkpoint_path, str):
        raise ValueError("checkpoint_path must be a non-empty string")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict: Dict[str, torch.Tensor]
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        # might already be a raw state dict
        state_dict = ckpt  # type: ignore[assignment]
    else:
        raise ValueError("Unsupported checkpoint format (expected dict)")

    base = _get_base_module(model)
    model_sd = base.state_dict()

    cleaned: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        nk = _normalize_ckpt_key(k)
        target_key = None
        for key_option in _possible_target_keys(nk):
            if key_option not in model_sd:
                continue
            if shape_filter and hasattr(v, "shape") and model_sd[key_option].shape != v.shape:
                continue
            target_key = key_option
            break
        if target_key is None:
            continue
        cleaned[target_key] = v

    base.load_state_dict(cleaned, strict=strict)
    return len(cleaned), len(state_dict)


def reset_classifier_head(model: nn.Module) -> bool:
    """
    Reset head weights if a known head module exists.
    Returns True if something was reset.
    """
    base = _get_base_module(model)
    head = None
    if hasattr(base, "head") and isinstance(getattr(base, "head"), nn.Module):
        head = getattr(base, "head")
    elif hasattr(base, "classifier") and isinstance(getattr(base, "classifier"), nn.Module):
        head = getattr(base, "classifier")

    if head is not None:
        for m in head.modules():
            if isinstance(m, nn.Linear) and hasattr(m, "reset_parameters"):
                m.reset_parameters()
        return True

    # fallback: reset last Linear
    last_linear = None
    for m in base.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is not None and hasattr(last_linear, "reset_parameters"):
        last_linear.reset_parameters()
        return True
    return False


def freeze_backbone_train_head(model: nn.Module, train_attnpool: bool = False) -> Tuple[int, int]:
    """
    Freeze all parameters, then unfreeze head/classifier if present.
    Returns (trainable_params, total_params).
    """
    base = _get_base_module(model)
    for p in base.parameters():
        p.requires_grad = False

    unfroze = False
    if hasattr(base, "head") and isinstance(getattr(base, "head"), nn.Module):
        for p in getattr(base, "head").parameters():
            p.requires_grad = True
        unfroze = True
    if not unfroze and hasattr(base, "classifier") and isinstance(getattr(base, "classifier"), nn.Module):
        for p in getattr(base, "classifier").parameters():
            p.requires_grad = True
        unfroze = True

    # Optional: also train pooling when freezing the backbone (useful for map-style pretraining)
    if train_attnpool and hasattr(base, "attnpool") and isinstance(getattr(base, "attnpool"), nn.Module):
        for p in getattr(base, "attnpool").parameters():
            p.requires_grad = True
    if not unfroze:
        # fallback: unfreeze last Linear
        last_linear = None
        for m in base.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        if last_linear is not None:
            for p in last_linear.parameters():
                p.requires_grad = True

    total = sum(p.numel() for p in base.parameters())
    trainable = sum(p.numel() for p in base.parameters() if p.requires_grad)
    return trainable, total
