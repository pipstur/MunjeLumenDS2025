import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".pre-commit-config.yaml", ".git", ".github"],
    pythonpath=True,
    dotenv=True,
)

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import hydra
import lightning as pl
import pandas as pd
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from training.src.train_utils import pylogger, train_utils
from training.src.train_utils.train_utils import avg_metrics, convert_tensor_to_float, sum_metrics

log = pylogger.get_pylogger(__name__)


@train_utils.task_wrapper
def train_kfold(cfg: DictConfig, n_splits: int = 5) -> Tuple[dict, dict]:
    """K-Fold Cross Validation Training."""
    # set seed for reproducibility
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Starting {n_splits}-Fold Cross Validation")

    # Retrieve full dataset and targets
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    fold_metrics = []  # To store metrics for each fold
    metrics_sum = {}
    datadir = datamodule.data_dir
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for fold in range(1, n_splits):
        datamodule.data_dir = f"{datadir}/fold{fold}"

        # Instantiate model, trainer, and other components
        log.info(f"Instantiating model <{cfg.model._target_}>")
        model: LightningModule = hydra.utils.instantiate(cfg.model)

        log.info("Instantiating callbacks...")
        callbacks: List[Callback] = train_utils.instantiate_callbacks(cfg.get("callbacks"))

        log.info("Instantiating loggers...")
        if OmegaConf.select(cfg, "logger") and OmegaConf.select(cfg.logger, "tensorboard"):
            cfg.get("logger").tensorboard.version = "fold" + str(fold)
        logger: List[Logger] = train_utils.instantiate_loggers(cfg.get("logger"))

        log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
        trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

        object_dict = {
            "cfg": cfg,
            "datamodule": datamodule,
            "model": model,
            "callbacks": callbacks,
            "logger": logger,
            "trainer": trainer,
        }

        if logger:
            log.info("Logging hyperparameters!")
            train_utils.log_hyperparameters(object_dict)

        model_checkpoint: ModelCheckpoint = callbacks[0]
        model_checkpoint.dirpath = model_checkpoint.dirpath + "/fold" + str(fold)

        if cfg.get("train"):
            log.info(f"Training Fold {fold + 1}")
            trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

        train_metrics = trainer.callback_metrics

        log.info(f"Evaluating Fold {fold + 1}")
        if cfg.get("test"):
            log.info("Starting testing!")
            ckpt_path = trainer.checkpoint_callback.best_model_path
            if ckpt_path == "":
                log.warning("Best ckpt not found! Using current weights for testing...")
                ckpt_path = None
            trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
            log.info(f"Best ckpt path: {ckpt_path}")

        test_metrics = trainer.callback_metrics

        metric_dict = {**train_metrics, **test_metrics}

        fold_metrics.append(
            {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in test_metrics.items()}
        )

        convert_tensor_to_float(metric_dict)
        metrics_sum = sum_metrics(metrics_sum, metric_dict)

    average_metrics = pd.DataFrame(fold_metrics).mean().to_dict()
    log.info(f"Average Metrics: {average_metrics}")
    metrics_avg = avg_metrics(metrics_sum, n_splits)

    metrics_csv_path = Path(model_checkpoint.dirpath) / f"kfold_metrics_{timestamp}.csv"
    metrics_data = pd.DataFrame(fold_metrics)
    metrics_data = metrics_data.append(average_metrics, ignore_index=True)
    metrics_data.to_csv(metrics_csv_path, index=False)
    log.info(f"Metrics saved to {metrics_csv_path}")

    return metrics_avg, object_dict


@hydra.main(
    version_base="1.3.2", config_path=str(root / "training" / "configs"), config_name="train.yaml"
)
def main(cfg: DictConfig) -> Optional[float]:
    # train the model with K-Fold
    metric_dict, _ = train_kfold(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = train_utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
