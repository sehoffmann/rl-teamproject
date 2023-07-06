import torch
from torch import nn

from .trainer import BaseTrainer


class ClassificationTrainer(BaseTrainer):
    def forward_step(self, batch_idx, batch):
        X, label = (tensor.to(self.device, non_blocking=True) for tensor in batch)
        pred = self.model(X)

        with torch.no_grad():
            acc = (pred.argmax(dim=1) == label).float().mean()
            self.log_metric('acc', acc)

            top5_indices = pred.topk(5, dim=1)[1]
            top5_error = 1 - (top5_indices == label.unsqueeze(1)).float().max(dim=1)[0].mean()
            self.log_metric('top5_error', top5_error)

        return self.loss_fn(pred, label)

    def create_loss(self):
        return nn.CrossEntropyLoss()

    def metric_names(self):
        return ['train/acc', 'val/acc', 'val/top5_error']
