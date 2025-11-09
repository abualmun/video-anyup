# Adapted from JAFAR (https://github.com/PaulCouairon/JAFAR)
import torch
from einops import rearrange
from torch import nn


class Cosine_MSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = torch.nn.MSELoss()
        self.cosine_loss = torch.nn.CosineEmbeddingLoss()

    def forward(self, pred, target):
        pred = rearrange(pred, "b c h w -> (b h w) c")
        target = rearrange(target, "b c h w -> (b h w) c")

        gt = torch.ones_like(target[:, 0])

        min_val = torch.min(target, dim=1, keepdim=True).values
        max_val = torch.max(target, dim=1, keepdim=True).values
        pred_normalized = (pred - min_val) / (max_val - min_val + 1e-6)
        target_normalized = (target - min_val) / (max_val - min_val + 1e-6)

        loss = self.cosine_loss(pred, target, gt) + self.mse_loss(pred_normalized, target_normalized)
        return {"total": loss}
