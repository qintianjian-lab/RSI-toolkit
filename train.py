"""
Downstream classifier training for the RSI paper pipeline.

Examples:
  python train.py --used_model sbm_universal_v2 --cross_name fold_1 --skip_test

  python train.py --used_model sbm_universal_v2 --cross_name fold_1 \
    --pretrained_ckpt runs/checkpoints_masked_pretrain/best.ckpt \
    --freeze_backbone --train_attnpool --reset_head --skip_test

For transfer from an in-domain masked-reconstruction or source-domain
checkpoint, optionally reset the classifier head and freeze the feature
extractor for a head-only warmup.
"""
import argparse
import os
from datetime import datetime

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from config.config import config
from config.model_config import get_model_kwargs
from dataset.dataset import build_dataloader
from model.lightning import BuildLightningModel
from utils.tools import set_random_seed, check_model_name
from utils.transfer_learning import load_pretrained_weights, freeze_backbone_train_head, reset_classifier_head


def train(
        used_model: str = '',
        cross_validation_fold_name: str = '',
        pretrained_ckpt: str = '',
        freeze_backbone: bool = False,
        train_attnpool: bool = False,
        reset_head: bool = False,
        strict_load: bool = False,
        skip_test: bool = False,
):
    torch.set_float32_matmul_precision('high')
    set_random_seed(config['random_seed'])

    if used_model != '' and used_model is not None:
        config['used_model'] = used_model
    check_model_name(config['used_model'])

    # devices setting
    precision = config['precision']
    used_device = config['used_device']

    # load dataset
    train_dataloader = build_dataloader(config, mode='train', cross_val_name=cross_validation_fold_name)
    val_dataloader = build_dataloader(config, mode='val', cross_val_name=cross_validation_fold_name)

    # model init
    model = BuildLightningModel(
        model_name=config['used_model'],
        learn_rate=config['learn_rate'],
        cos_annealing_t_0=config['cos_annealing_t_0'],
        cos_annealing_t_mult=config['cos_annealing_t_mult'],
        cos_annealing_eta_min=config['cos_annealing_eta_min'],
        weight_decay=config.get('weight_decay', 1e-2),
        in_channel=config['in_channel'],
        spectrum_size=config['spectrum_size'],
        num_classes=len(config['type_list']),
        classes_name_list=config['type_list'],
        enable_torch_2=config['enable_torch_2.0'],
        torch_2_compile_mode=config['torch_2.0_compile_mode'],
        model_kwargs=get_model_kwargs(config['used_model'], config),
    )

    # ===== Transfer Learning (opt-in; does NOT affect default training) =====
    if isinstance(pretrained_ckpt, str) and pretrained_ckpt.strip():
        print(f"[TL] Loading pretrained checkpoint: {pretrained_ckpt}")
        loaded, total = load_pretrained_weights(model.model, pretrained_ckpt, strict=strict_load, shape_filter=True)
        print(f"[TL] Loaded keys: {loaded}/{total} (shape-matched)")
        if loaded <= 0:
            print(
                "[TL][Warning] 0 keys loaded. This usually means the checkpoint parameter names/prefixes or "
                "architecture hyperparameters do not match the current model.\n"
                "  - If this ckpt comes from wrapper pretraining, keys may be under encoder.* or encoder.backbone.* and must map to the downstream backbone keys.\n"
                "  - Ensure width/shape-related args match (e.g., base_channels/stage_depths/stem_type/block_type and classifier head shape).\n"
                "  - If you still see 0 after updating utils/transfer_learning.py, share a few state_dict keys from the ckpt for inspection."
            )
        if reset_head:
            did = reset_classifier_head(model.model)
            print(f"[TL] Reset head: {did}")
        if freeze_backbone:
            trainable, total_params = freeze_backbone_train_head(model.model, train_attnpool=train_attnpool)
            print(f"[TL] Freeze backbone enabled. Trainable params: {trainable}/{total_params}")

    # log settings
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    print('[Info] Training start time: ', current_time)
    logger_list = []
    tensorboard_logger = TensorBoardLogger(save_dir=config['log_dir'], name='{}'.format(current_time))
    tensorboard_logger.log_hyperparams(config)
    logger_list.append(tensorboard_logger)
    if not config['debug'] and config['enable_wandb']:
        wandb_logger = WandbLogger(project=config['wandb_project_name'], save_dir=config['log_dir'],
                                   name='{}-{}'.format(config['used_model'], current_time))
        logger_list.append(wandb_logger)

    # early stopping
    monitor_metric = config.get('monitor_metric', 'val_acc')
    monitor_mode = config.get('monitor_mode', 'max')
    early_stop_callback = EarlyStopping(monitor_metric, mode=monitor_mode, min_delta=0.005, patience=40, verbose=True)

    # make checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config['checkpoint_dir'], '{}'.format(current_time), 'checkpoints'),
        filename=f"best-{{epoch}}-{{{monitor_metric}:.5f}}",
        save_top_k=5,
        monitor=monitor_metric,
        mode=monitor_mode,
        save_weights_only=False
    )

    # lr monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # init trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=used_device,
        precision=precision,
        logger=logger_list,
        callbacks=[checkpoint_callback, lr_monitor, early_stop_callback],
        max_epochs=config['epochs'],
        log_every_n_steps=1,
        enable_progress_bar=True,
        check_val_every_n_epoch=1,
        fast_dev_run=config['debug'],
        # inference_mode=False,  # to avoid pl test error when using torch 2.0
    )
    # train
    trainer.fit(model, train_dataloader, val_dataloader)

    # test
    best_model_path = checkpoint_callback.best_model_path
    print('[Info] best model path: ', best_model_path)
    if skip_test:
        print('[Info] skip_test enabled. Skip test.')
    else:
        best_model = BuildLightningModel.load_from_checkpoint(best_model_path)
        best_model.eval()
        # Optional test: skip if test split missing or empty
        try:
            test_dataloader = build_dataloader(config, mode='test', cross_val_name=cross_validation_fold_name)
            dataset_len = len(getattr(test_dataloader, 'dataset', []))
            if dataset_len <= 0:
                print('[Info] No test samples found. Skip test.')
            else:
                trainer.test(best_model, test_dataloader)
        except Exception as e:
            print(f'[Info] Skip test due to missing/invalid test split: {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--used_model', type=str, default='')
    parser.add_argument('--cross_name', type=str, default='')
    # transfer learning (opt-in)
    parser.add_argument('--pretrained_ckpt', type=str, default='', help='Path to pretrained ckpt (e.g., SDSS).')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze feature extractor; train only head.')
    parser.add_argument('--train_attnpool', action='store_true',
                        help='When used with --freeze_backbone, also train attnpool together with head (recommended for AE map pretrain).')
    parser.add_argument('--reset_head', action='store_true', help='Reset classifier head parameters after loading.')
    parser.add_argument('--strict_load', action='store_true', help='Strict state_dict loading (not recommended for TL).')
    parser.add_argument('--skip_test', action='store_true', help='Skip automatic test after training.')
    # optional overrides
    parser.add_argument('--epochs', type=int, default=-1, help='Override config[epochs] for this run only.')
    parser.add_argument('--learn_rate', type=float, default=-1.0, help='Override config[learn_rate] for this run only.')
    parser.add_argument('--weight_decay', type=float, default=-1.0, help='Override config[weight_decay] for this run only.')
    parser.add_argument('--monitor_metric', type=str, default='', help='Override config[monitor_metric] for this run only.')
    parser.add_argument('--monitor_mode', type=str, default='', help='Override config[monitor_mode] for this run only.')
    parser.add_argument('--random_seed', type=int, default=-1, help='Override config[random_seed] for this run only.')
    parser.add_argument('--dataset_dir', type=str, default='', help='Override config[dataset_dir] for this run only.')
    parser.add_argument('--spectrum_size', type=int, default=-1, help='Override config[spectrum_size] for this run only.')
    opt = parser.parse_args()
    # apply optional overrides before training
    if isinstance(opt.random_seed, int) and opt.random_seed > -1:
        config['random_seed'] = int(opt.random_seed)
    if isinstance(opt.epochs, int) and opt.epochs > 0:
        config['epochs'] = int(opt.epochs)
    if isinstance(opt.learn_rate, float) and opt.learn_rate > 0:
        config['learn_rate'] = float(opt.learn_rate)
    if isinstance(opt.weight_decay, float) and opt.weight_decay >= 0:
        config['weight_decay'] = float(opt.weight_decay)
    if opt.monitor_metric.strip():
        config['monitor_metric'] = opt.monitor_metric.strip()
    if opt.monitor_mode.strip():
        config['monitor_mode'] = opt.monitor_mode.strip()
    if opt.dataset_dir.strip():
        config['dataset_dir'] = opt.dataset_dir.strip()
    if isinstance(opt.spectrum_size, int) and opt.spectrum_size > 0:
        config['spectrum_size'] = int(opt.spectrum_size)
    train(
        opt.used_model,
        opt.cross_name,
        opt.pretrained_ckpt,
        opt.freeze_backbone,
        opt.train_attnpool,
        opt.reset_head,
        opt.strict_load,
        opt.skip_test,
    )
