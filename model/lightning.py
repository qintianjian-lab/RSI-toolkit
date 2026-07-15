from __future__ import annotations

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassRecall,
    MulticlassPrecision,
    MulticlassAUROC,
    BinaryAveragePrecision,
    MulticlassAveragePrecision,
)

from utils.tools import check_model_name

try:
    from model.mspc_net import MSPCNET
    from model.sbm_universal import SBM_Universal_Framework as SBM_UNIVERSAL
    from model.sbm_universal_v2 import SBM_Universal_Framework as SBM_UNIVERSAL_V2
except ImportError as exc:
    raise ImportError(f'[Error] import model failed: {exc}') from exc


class BuildLightningModel(pl.LightningModule):
    def __init__(self,
                 model_name: str,
                 learn_rate: float,
                 cos_annealing_t_0: int,
                 cos_annealing_t_mult: int,
                 cos_annealing_eta_min: float,
                 in_channel: int,
                 spectrum_size: int,
                 num_classes: int,
                 classes_name_list: list[str, ...],
                 label_smoothing: float = 0.,
                 enable_torch_2: bool = True,
                 torch_2_compile_mode: str = 'default',
                 weight_decay: float = 0.0,
                 model_kwargs: dict | None = None,
                 **_unused_checkpoint_kwargs
                 ):
        super().__init__()
        assert torch_2_compile_mode in ['default', 'reduce-overhead', 'max-autotune'], \
            '[Error] torch_2_compile_mode must be in [default, reduce-overhead, max-autotune]'
        check_model_name(model_name)
        print('[Info] model_name: {}'.format(model_name))

        model_kwargs = model_kwargs or {}
        self.model = eval(model_name.upper())(
            in_channel=in_channel,
            out_channel=num_classes,
            spectrum_size=spectrum_size,
            **model_kwargs
        ) if not enable_torch_2 else torch.compile(
            eval(model_name.upper())(
                in_channel=in_channel,
                out_channel=num_classes,
                spectrum_size=spectrum_size,
                **model_kwargs
            ),
            mode=torch_2_compile_mode
        )
        if enable_torch_2:
            print('[Info] Using PyTorch 2.0 compile')
        self.learn_rate = learn_rate
        self.cos_annealing_t_0 = cos_annealing_t_0
        self.cos_annealing_t_mult = cos_annealing_t_mult
        self.cos_annealing_eta_min = cos_annealing_eta_min
        self.weight_decay = weight_decay
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.num_classes = num_classes

        self.train_acc = MulticlassAccuracy(num_classes=num_classes).to(self.device)
        self.train_precision = MulticlassPrecision(num_classes=num_classes).to(self.device)
        self.train_recall = MulticlassRecall(num_classes=num_classes).to(self.device)
        self.train_auroc = MulticlassAUROC(num_classes=num_classes).to(self.device)

        self.val_acc = MulticlassAccuracy(num_classes=num_classes).to(self.device)
        self.val_precision = MulticlassPrecision(num_classes=num_classes).to(self.device)
        self.val_recall = MulticlassRecall(num_classes=num_classes).to(self.device)
        self.val_auroc = MulticlassAUROC(num_classes=num_classes).to(self.device)
        # AUPRC is used for monitoring/early stopping.
        if num_classes == 2:
            self.val_auprc = BinaryAveragePrecision().to(self.device)
        else:
            self.val_auprc = MulticlassAveragePrecision(num_classes=num_classes, average='macro').to(self.device)

        self.test_acc = MulticlassAccuracy(num_classes=num_classes).to(self.device)
        self.test_precision = MulticlassPrecision(num_classes=num_classes).to(self.device)
        self.test_recall = MulticlassRecall(num_classes=num_classes).to(self.device)
        self.test_auroc = MulticlassAUROC(num_classes=num_classes).to(self.device)
        self.test_pred = []
        self.test_label = []
        self.classes_name_list = classes_name_list

        self.softmax = nn.Softmax(dim=1)
        self.best_val_acc = 0.0
        self.save_hyperparameters(ignore=["_unused_checkpoint_kwargs"])

    def training_step(self, batch, batch_idx):
        spectrum, label, meta = batch

        if hasattr(self.model, 'forward_with_context'):
            pred = self.model.forward_with_context(spectrum, meta)
            loss = self.criterion(pred, label)
        else:
            pred = self.model(spectrum)
            loss = self.criterion(pred, label)

        base_model = getattr(self.model, "_orig_mod", self.model)
        if hasattr(base_model, "cache_with_labels"):
            try:
                base_model.cache_with_labels(spectrum, label)
            except Exception:
                pass

        self.train_acc(pred, label)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=True, batch_size=spectrum.shape[0])
        self.train_precision(pred, label)
        self.log('train_precision', self.train_precision, prog_bar=True, on_step=True, batch_size=spectrum.shape[0])
        self.train_recall(pred, label)
        self.log('train_recall', self.train_recall, prog_bar=True, on_step=True, batch_size=spectrum.shape[0])
        self.train_auroc(pred, label)
        self.log('train_auroc', self.train_auroc, prog_bar=True, on_step=True, batch_size=spectrum.shape[0])

        self.log('train_loss', loss, prog_bar=True, on_step=True, batch_size=spectrum.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        spectrum, label, meta = batch

        if hasattr(self.model, 'forward_with_context'):
            pred = self.model.forward_with_context(spectrum, meta)
        else:
            pred = self.model(spectrum)
        loss = self.criterion(pred, label)

        self.val_acc(pred, label)
        self.val_precision(pred, label)
        self.val_recall(pred, label)
        self.val_auroc(pred, label)
        probs = self.softmax(pred)
        if self.num_classes == 2:
            self.val_auprc(probs[:, 1], label)
        else:
            self.val_auprc(probs, label)

        self.log('val_loss', loss, prog_bar=True, on_step=True, batch_size=spectrum.shape[0])
        return loss

    def on_validation_epoch_end(self):
        _val_acc = self.val_acc.compute()
        if _val_acc > self.best_val_acc:
            self.best_val_acc = _val_acc
            self.log('best_val_acc', self.best_val_acc, prog_bar=True, on_epoch=True)
        self.log('val_acc', _val_acc, prog_bar=True, on_epoch=True)
        self.val_acc.reset()

        _val_precision = self.val_precision.compute()
        self.log('val_precision', _val_precision, prog_bar=True, on_epoch=True)
        self.val_precision.reset()

        _val_recall = self.val_recall.compute()
        self.log('val_recall', _val_recall, prog_bar=True, on_epoch=True)
        self.val_recall.reset()

        _val_auroc = self.val_auroc.compute()
        self.log('val_auroc', _val_auroc, prog_bar=True, on_epoch=True)
        self.val_auroc.reset()
        # Log and reset AUPRC.
        _val_auprc = self.val_auprc.compute()
        self.log('val_auprc', _val_auprc, prog_bar=True, on_epoch=True)
        self.val_auprc.reset()

    def test_step(self, batch, batch_idx):
        import time
        start_time = time.time()
        
        spectrum, label, meta = batch

        if hasattr(self.model, 'forward_with_context'):
            pred = self.model.forward_with_context(spectrum, meta)
        else:
            pred = self.model(spectrum)
        loss = self.criterion(pred, label)

        self.test_acc(pred, label)
        self.test_precision(pred, label)
        self.test_recall(pred, label)
        self.test_auroc(pred, label)

        self.test_pred.extend(list(np.argmax(self.softmax(pred).cpu().numpy(), axis=1)))
        self.test_label.extend(list(label.cpu().numpy()))

        # Log inference time.
        inference_time = time.time() - start_time
        self.log('test_inference_time', inference_time, prog_bar=True, on_step=True, batch_size=spectrum.shape[0])
        
        self.log('test_loss', loss, prog_bar=True, on_step=True, batch_size=spectrum.shape[0])
        return loss

    def on_test_epoch_end(self):
        _test_acc = self.test_acc.compute()
        self.log('test_acc', _test_acc, prog_bar=True, on_epoch=True)
        self.test_acc.reset()

        _test_precision = self.test_precision.compute()
        self.log('test_precision', _test_precision, prog_bar=True, on_epoch=True)
        self.test_precision.reset()

        _test_recall = self.test_recall.compute()
        self.log('test_recall', _test_recall, prog_bar=True, on_epoch=True)
        self.test_recall.reset()

        _test_auroc = self.test_auroc.compute()
        self.log('test_auroc', _test_auroc, prog_bar=True, on_epoch=True)
        self.test_auroc.reset()

        # Log confusion matrix only when WandB is active.
        if wandb.run is not None:
            cm = wandb.plot.confusion_matrix(
                preds=np.asarray(self.test_pred),
                y_true=np.asarray(self.test_label),
                probs=None,
                class_names=self.classes_name_list,
                title="Confusion Matrix in Test dataset",
            )
            wandb.log({"Confusion Matrix": cm})

    def configure_optimizers(self):
        # IMPORTANT: keep default behavior unchanged when all params are trainable.
        # This also supports transfer-learning where most params are frozen.
        # AdamW with weight decay is a stable default for small data and large models.
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=self.learn_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.cos_annealing_t_0,
            T_mult=self.cos_annealing_t_mult,
            eta_min=self.cos_annealing_eta_min
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'name': 'lr'
            },
        }
