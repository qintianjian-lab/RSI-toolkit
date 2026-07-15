"""
Per-model constructor kwargs.

Training/data settings live in config.py; this module only defines model
constructor arguments. Runtime overrides can be supplied through
config["model_kwargs_<used_model>"].
"""
from typing import Any, Dict, Optional

# Model constructor defaults.
MODEL_CONFIG: Dict[str, Dict[str, Any]] = {
    # Non-SBM reference model.
    "mspcnet": {
        "embed_dim": 128,
        "depth": [4, 4, 8, 4],
        "depth_scale": 1.0,
        "mp_ratio": 2.0,
        "n_div": 16,
        "patch_conv_size": 3,
        "patch_conv_stride": 1,
        "pc_conv_size": 5,
        "pc_conv_size_scale": 9.0,
        "merge_conv_size": 3,
        "merge_conv_stride": 2,
        "head_dim": 1024,
        "drop_path_rate": 0.3,
        "norm_layer": "BN",
        "act_layer": "GELU",
        "attention": "ECA_F",
    },

    # RSI/SBM models included in this toolkit.
    "sbm_universal": {
        # stem_type:
        # - "mspc"        : PatchEmbed-aligned MSPC stem
        # - "mspc_legacy" : Conv5/2+BN+ReLU stem used by existing checkpoints
        "block_type": "mspc",
        "stem_type": "mspc_legacy",
        "preband_split": True,
        "stage_depths": [2, 2, 2, 1],
        "base_channels": 48,
        "drop_path": 0.1,
        "k_low": 65,
        "k_mid": 17,
        # only used when stem_type contains "mspc" (string or per-band dict)
        # align with mspc_net PatchEmbed
        "mspc_stem_kwargs": {
            "patch_conv_size": 3,
            "patch_conv_stride": 1,
            "norm_layer": "BN",
            "act_layer": "NONE",
        },
        # only used when block_type contains "mspc" (string or per-band dict)
        "mspc_block_kwargs": {
            "n_div": 16,
            "mp_ratio": 2.0,
            "pc_conv_size": 5,
            "pc_conv_size_scale": 9.0,
            "norm_layer": "BN",
            "act_layer": "GELU",
            "attention": "ECA_F",
        },
    },
    "sbm_universal_v2": {
        "block_type": "mspc",
        "stem_type": "mspc_legacy",
        "preband_split": True,
        "stage_depths": [2, 2, 2, 1],
        "base_channels": 48,
        "drop_path": 0.1,
        "k_low": 65,
        "k_mid": 17,
        "mspc_stem_kwargs": {
            "patch_conv_size": 3,
            "patch_conv_stride": 1,
            "norm_layer": "BN",
            "act_layer": "NONE",
        },
        "mspc_block_kwargs": {
            "n_div": 16,
            "mp_ratio": 2.0,
            "pc_conv_size": 5,
            "pc_conv_size_scale": 9.0,
            "norm_layer": "BN",
            "act_layer": "GELU",
            "attention": "ECA_F",
        },
        # Main v2 configuration uses the Stage-4 Transformer mixer.
        "global_mixer": "transformer",
        "global_mixer_depth": 1,
        "global_attn_heads": 4,
        "global_pos_encoding": "sincos",
        "global_learned_pos_max_len": 512,
        "global_mixer_insert_after": "stage4",
    },
}


def get_model_kwargs(model_name: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Return constructor kwargs for a model name.

    - If config contains model_kwargs_<model_name>, merge it over the defaults.
    - Otherwise return MODEL_CONFIG defaults.
    """
    out = dict(MODEL_CONFIG.get(model_name, {}))
    if config is not None:
        override = config.get(f"model_kwargs_{model_name}")
        if isinstance(override, dict) and override:
            out.update(override)
    return out
