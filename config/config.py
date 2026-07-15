import os


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


config = {
    # Runtime
    'debug': False,
    'wandb_project_name': 'RSI_PSB_Classification',
    'enable_wandb': False,
    'random_seed': 42,
    'used_device': [0],
    'precision': '16-mixed',

    # Dataset layout:
    #   dataset_dir/fold_1/{train,val,test}/{spectrum,label}/...
    # Override with --dataset_dir or RSI_DATASET_DIR.
    'dataset_dir': os.environ.get('RSI_DATASET_DIR', os.path.join(REPO_ROOT, 'data', 'lamost_folds')),
    'spectrum_dir': 'spectrum',
    'label_dir': 'label',
    'type_list': [0, 1],

    # Example runtime default; override for specific experiments.
    'used_model': 'sbm_universal_v2',
    'enable_torch_2.0': False,
    'torch_2.0_compile_mode': 'default',

    # Downstream training defaults.
    'in_channel': 1,
    'spectrum_size': 4000,
    'batch_size': 32,
    'val_batch_size': 256,
    'num_workers': 4,
    'epochs': 100,
    'learn_rate': 1e-4,
    'weight_decay': 1e-2,
    'cos_annealing_t_0': 20,
    'cos_annealing_t_mult': 2,
    'cos_annealing_eta_min': 1e-6,

    'log_dir': os.environ.get('RSI_LOG_DIR', os.path.join(REPO_ROOT, 'runs', 'logs')),
    'checkpoint_dir': os.environ.get('RSI_CKPT_DIR', os.path.join(REPO_ROOT, 'runs', 'checkpoints')),

    # Monitor minority-class ranking quality by default.
    'monitor_metric': 'val_auprc',
    'monitor_mode': 'max',
}
