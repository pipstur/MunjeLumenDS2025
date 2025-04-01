import torch
from torch import nn
from torchvision.models import SqueezeNet1_1_Weights, squeezenet1_1

from training.src.models.components.loss_functions import FocalLoss
from training.src.models.components.model_class import Model

torch.use_deterministic_algorithms(True, warn_only=True)


class SqueezeNet1_1(Model):
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
        loss_function: str = "cross_entropy",
    ):
        super().__init__()

        # loss function
        if loss_function == "cross_entropy":
            self.criterion = torch.nn.CrossEntropyLoss()
        elif loss_function == "focal":
            self.criterion = FocalLoss(alpha=[0.1, 0.9], gamma=2.0)

        backbone = squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)
        num_filters = backbone.classifier[1].in_channels

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
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "squeezenet1_1.yaml")
    _ = hydra.utils.instantiate(cfg)
