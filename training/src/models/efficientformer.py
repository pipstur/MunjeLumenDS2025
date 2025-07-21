import timm
import torch
from torch import nn

from training.src.models.components.loss_functions import get_loss_function
from training.src.models.components.model_class import Model

torch.use_deterministic_algorithms(True, warn_only=True)


class EfficientFormer(Model):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        freeze_layers: bool,
        loss_function: str = "cross_entropy",
    ):
        super().__init__()

        self.criterion = get_loss_function(loss_function)

        # Use num_classes=0 to remove the default classifier
        backbone = timm.create_model("efficientformer_l1", pretrained=True, num_classes=0)
        num_filters = backbone.num_features

        self.feature_extractor = backbone

        if freeze_layers:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        self.classifier = nn.Linear(num_filters, self.num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "efficientformer.yaml")
    _ = hydra.utils.instantiate(cfg)
