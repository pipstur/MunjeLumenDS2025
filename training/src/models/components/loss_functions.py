import torch


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=[0.2, 0.8], gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha) if isinstance(alpha, (list, tuple)) else alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        targets = torch.nn.functional.one_hot(targets, num_classes=2).float()
        inputs = inputs.to(targets.device)  # Ensure matching device

        BCE_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        pt = torch.exp(-BCE_loss)

        alpha = self.alpha.to(inputs.device)  # Ensure alpha is on the correct device
        alpha_t = alpha[targets.argmax(dim=1)].unsqueeze(1)

        loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss
        return loss.mean() if self.reduction == "mean" else loss.sum()
