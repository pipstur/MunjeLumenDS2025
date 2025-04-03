import io
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision
from lightning import LightningModule
from PIL import Image
from torchmetrics import (
    AUROC,
    Accuracy,
    AveragePrecision,
    ConfusionMatrix,
    F1Score,
    MaxMetric,
    MeanMetric,
    Precision,
    Recall,
    Specificity,
)

torch.use_deterministic_algorithms(True, warn_only=True)


class Model(LightningModule):
    """Implementation of LightningModule.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.num_classes = 2

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")

        self.train_spec = Specificity(task="binary")
        self.val_spec = Specificity(task="binary")
        self.test_spec = Specificity(task="binary")

        self.train_f1 = F1Score(task="binary")
        self.val_f1 = F1Score(task="binary")
        self.test_f1 = F1Score(task="binary")

        self.train_recall = Recall(task="binary")
        self.val_recall = Recall(task="binary")
        self.test_recall = Recall(task="binary")

        self.train_precision = Precision(task="binary")
        self.val_precision = Precision(task="binary")
        self.test_precision = Precision(task="binary")

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.con_mat = ConfusionMatrix(task="binary", num_classes=self.num_classes)

        self.train_roc_auc = AUROC(task="binary")
        self.val_roc_auc = AUROC(task="binary")
        self.test_roc_auc = AUROC(task="binary")

        self.train_auprc = AveragePrecision(task="binary")
        self.val_auprc = AveragePrecision(task="binary")
        self.test_auprc = AveragePrecision(task="binary")

        # for tracking best so far validation accuracy
        self.val_roc_auc_best = MaxMetric()
        self.feature_extractor = None
        self.classifier = None

    def forward(self, x):
        features = self.feature_extractor(x)

        features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))

        features = features.view(features.size(0), -1)

        # Pass through the classifier
        output = self.classifier(features)
        return output

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_roc_auc_best doesn't store accuracy from these checks
        self.val_roc_auc_best.reset()

    def model_step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        probs = torch.softmax(logits, dim=1)[:, 1]
        return loss, preds, y, probs

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, probs = self.model_step(batch)
        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.train_spec(preds, targets)
        self.train_recall(preds, targets)
        self.train_precision(preds, targets)
        self.train_f1(preds, targets)
        self.train_roc_auc(preds, targets)
        self.train_auprc(probs, targets)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/spec", self.train_spec, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/recall", self.train_recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/prec", self.train_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/roc_auc", self.train_roc_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/auprc", self.train_auprc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `on_train_epoch_end` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def on_train_epoch_end(self):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, probs = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_spec(preds, targets)
        self.val_recall(preds, targets)
        self.val_precision(preds, targets)
        self.val_f1(preds, targets)
        self.val_roc_auc(preds, targets)
        self.val_auprc(probs, targets)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/spec", self.val_spec, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recall", self.val_recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/prec", self.val_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/roc_auc", self.val_roc_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auprc", self.val_auprc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def on_validation_epoch_end(self):
        roc_auc = self.val_roc_auc.compute()  # get current val acc
        self.val_roc_auc_best(roc_auc)  # update best so far val acc
        self.log("val/roc_auc_best", self.val_roc_auc_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, probs = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.test_spec(preds, targets)
        self.test_recall(preds, targets)
        self.test_precision(preds, targets)
        self.test_f1(preds, targets)
        self.con_mat(preds, targets)
        self.test_roc_auc(preds, targets)
        self.test_auprc(probs, targets)

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/spec", self.test_spec, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/recall", self.test_recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/prec", self.test_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/roc_auc", self.test_roc_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/auprc", self.test_auprc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def predict_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        return {"preds": preds, "targets": targets}

    def on_test_epoch_end(self):
        classes = ["Benign", "Malignant"]

        confusion_matrix = self.con_mat.compute().cpu().numpy()
        sums = np.sum(confusion_matrix, axis=1, keepdims=True)
        sums[sums == 0] = 1

        normalized_cm = confusion_matrix / sums
        normalized_cm[sums.squeeze() == 1, :] = 0

        formatted_cm = np.array(
            [
                [
                    f"{int(confusion_matrix[i, j])} ({normalized_cm[i, j] * 100:.1f}%)"
                    for j in range(len(classes))
                ]
                for i in range(len(classes))
            ]
        )

        df_cm = pd.DataFrame(
            normalized_cm * 100,
            index=[f"Actual {cls}" for cls in classes],
            columns=[f"Predicted {cls}" for cls in classes],
        )

        plt.figure(figsize=(12, 7))
        sns.heatmap(df_cm, annot=formatted_cm, fmt="", cmap="Blues", cbar=True, vmin=0, vmax=100)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix (Counts & %)")

        buf = io.BytesIO()
        plt.savefig(buf, format="jpeg", bbox_inches="tight")
        buf.seek(0)

        cm_image = Image.open(buf)
        cm_image = torchvision.transforms.ToTensor()(cm_image)
        self.logger.experiment.add_image(
            "confusion_matrix", cm_image, global_step=self.current_epoch
        )

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
