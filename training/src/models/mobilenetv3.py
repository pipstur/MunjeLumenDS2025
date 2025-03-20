import torch
from torch import nn
from torchvision import models

from training.src.models.components.model_class import Model

torch.use_deterministic_algorithms(True, warn_only=True)


class MobileNetV3(Model):
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
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        freeze_layers: bool,
    ):
        super().__init__()

        backbone = models.mobilenet_v3_large(pretrained=True)
        num_filters = backbone.classifier[0].in_features

        self.feature_extractor = nn.Sequential(*list(backbone.features.children()))

        if freeze_layers:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        self.classifier = nn.Linear(num_filters, self.num_classes)


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "mobilenetv3.yaml")
    _ = hydra.utils.instantiate(cfg)
