import torch
from timm import create_model
from torch import nn

from training.src.models.components.loss_functions import get_loss_function
from training.src.models.components.model_class import Model

torch.use_deterministic_algorithms(True, warn_only=True)


class MobileViT(Model):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        freeze_layers: bool,
        loss_function: str = "cross_entropy",
    ):
        super().__init__()

        self.criterion = get_loss_function(loss_function)

        backbone = create_model(
            "mobilevit_s", pretrained=True, num_classes=0
        )  # num_classes=0 to get features only

        num_filters = backbone.head.in_features

        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        if freeze_layers:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # Replace classifier to fit your num_classes
        self.classifier = nn.Linear(num_filters, self.num_classes)


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "mobilevit.yaml")
    _ = hydra.utils.instantiate(cfg)
