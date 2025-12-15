import timm

from lightning.pytorch import LightningModule
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torchmetrics import Accuracy, AUROC
import torch
class Xception(LightningModule):
    def __init__(self, lr, distributed=False):
        super(Xception, self).__init__()
        self.lr = lr
        self.model = timm.create_model('xception', pretrained=True, num_classes=1)
        self.loss_fn = BCEWithLogitsLoss()
        self.distributed = distributed
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.val_auroc = AUROC(task="binary")
    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y.unsqueeze(1))
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        probs = torch.sigmoid(y_hat)
        self.train_acc.update(probs, y.unsqueeze(1))
        self.log('train_acc', self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y.unsqueeze(1))
        probs = torch.sigmoid(y_hat)
        self.val_acc.update(probs, y.unsqueeze(1))
        self.val_auroc.update(probs, y.unsqueeze(1))
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_acc', self.val_acc, prog_bar=True)
        self.log('val_auroc', self.val_auroc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return [optimizer]
