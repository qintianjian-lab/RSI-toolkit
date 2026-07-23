"""
Example runtime defaults for `python -m rsi pretrain-mae`; override paths and
hyperparameters for specific experiments.

Set RSI_MAE_TRAIN_INDEX and RSI_MAE_VAL_INDEX to CSV files with columns:
basename,npy_path. The npy files should contain preprocessed 4000-point spectra.
"""

import os


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

CFG = {
    "data": {
        "spectrum_size": 4000,
        # Required: preprocessed cached arrays index files
        "cached_train_index_csv": os.environ.get("RSI_MAE_TRAIN_INDEX", ""),
        "cached_val_index_csv": os.environ.get("RSI_MAE_VAL_INDEX", ""),
    },
    "model": {
        # Backward-compatible default: pre-train the common SBM stem/stage encoder.
        # Set encoder_model="sbm_universal_v2" and global_mixer="transformer" to
        # pre-train the full v2 encoder used by the main downstream classifier.
        "encoder_model": os.environ.get("RSI_MAE_ENCODER_MODEL", "sbm_universal"),
        "block_type": "mspc",
        "stem_type": "mspc_legacy",
        "preband_split": True,
        "stage_depths": [2, 2, 2, 1],
        "band_stage_depths": None,
        "base_channels": 48,
        "drop_path": 0.1,
        # align with mspc_net PatchEmbed
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
        "k_low": 65,
        "k_mid": 17,
        # Only used when encoder_model is sbm_universal_v2.
        "global_mixer": os.environ.get("RSI_MAE_GLOBAL_MIXER", "none"),
        "global_mixer_depth": 1,
        "global_attn_heads": 4,
        "global_pos_encoding": "sincos",
        "global_learned_pos_max_len": 512,
        "global_mixer_insert_after": "stage4",
        # Weak decoder
        "decoder_hidden": 128,
        "decoder_fullres_refine": True,
    },
    "mask": {
        "mask_ratio": 0.6,
        "mask_block": 30,
        "mask_fill": "mean",
        "mask_loss_only": True,
    },
    "physics_mask": {
        "enabled": False,
        "prob": 0.3, # 0.3
        "only": False,
        "lines": "4101.7,4340.5,4861.3",
        "line_width": 20.0,
        # Set explicit wavelengths if physics mask is enabled.
        "wl_min": 3700.0,
        "wl_max": 6800.0,
    },
    "loss": {
        "recon_loss": "smoothl1",  # mse|smoothl1
        "recon_weight": 1.0,
        "grad_weight": 0.0, # 0.05
        "grad_loss": "mse",
        "grad_mask_only": True,
    },
    "distill": {
        "enabled": False,
        "ema": 0.99,
        "weight": 1.0, # 0.3
        "loss": "mse",
    },
    "train": {
        "batch_size": 128,
        "num_workers": 8,
        "epochs": 60,
        "lr": 4e-4,
        "device": os.environ.get("RSI_MAE_DEVICE", "cuda:0"),
        # Precision for PyTorch Lightning Trainer.
        # Common choices: "32-true" (fp32), "16-mixed" (AMP fp16), "bf16-mixed" (AMP bf16, if supported).
        "precision": "16-mixed",
        "log_dir": os.environ.get("RSI_MAE_LOG_DIR", os.path.join(REPO_ROOT, "runs", "logs_umae")),
        "ckpt_dir": os.environ.get("RSI_MAE_CKPT_DIR", os.path.join(REPO_ROOT, "runs", "checkpoints_umae")),
        "val_check_interval": 1.0,
    },
    "logging": {
        "enable_wandb": False,
        "wandb_project": "SBM_Universal_MAE",
        "wandb_name": "sbm_universal_mae_distill_isolated_v1",
    },
}
