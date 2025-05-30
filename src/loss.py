import torch
import torch.nn as nn
import torch.nn.functional as F

class SymmetricCrossEntropyLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, num_classes=6, eps=1e-4):
        super(SymmetricCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, logits, targets):
        # logits: [batch_size, num_classes]
        # targets: [batch_size] (int64)

        # Cross entropy
        ce = F.cross_entropy(logits, targets)

        # Convert targets to one-hot
        one_hot = torch.zeros_like(logits).scatter(1, targets.view(-1, 1), 1)
        one_hot = one_hot * (1 - self.eps) + self.eps / self.num_classes  # label smoothing

        # Softmax over logits
        pred = F.softmax(logits, dim=1)

        # Reverse cross entropy
        rce = (-pred * torch.log(one_hot)).sum(dim=1).mean()

        loss = self.alpha * ce + self.beta * rce
        return loss
