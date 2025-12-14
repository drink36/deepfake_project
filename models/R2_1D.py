import lightning.pytorch as pl
from torch import nn
from torchvision.models import video as video_models
from torchmetrics import Accuracy, AUROC
import torch

class R2Plus1D(pl.LightningModule):
    """
    R(2+1)D model for video-level deepfake detection.
    Pretrained on Kinetics-400.
    """
    
    def __init__(
        self,
        learning_rate: float = 1e-5,
        weight_decay: float = 1e-5,
        num_classes: int = 1,
        pretrained: bool = True,
        distributed: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.distributed = distributed
        
        # Load pretrained R(2+1)D
        if pretrained:
            self.model = video_models.r2plus1d_18(
                weights=video_models.R2Plus1D_18_Weights.KINETICS400_V1
            )
        else:
            self.model = video_models.r2plus1d_18(weights=None)
        
        # Replace classifier head
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )
        mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1,1)
        std  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1,1)
        self.register_buffer("img_mean", mean, persistent=False)
        self.register_buffer("img_std", std, persistent=False)
        # Loss and metrics
        self.criterion = nn.BCEWithLogitsLoss()
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.val_auroc = AUROC(task="binary")
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.img_mean) / self.img_std
    def forward(self, x):
        x = self._normalize(x)
        # x: (B, C, T, H, W)
        return self.model(x) 
    def training_step(self, batch, batch_idx):
        x, y = batch[:2]                # x: (B, 3, T, H, W), y: (B,)
        logits = self(x).squeeze(-1)    # (B,)
        loss = self.criterion(logits, y.float())
        probs = torch.sigmoid(logits)
        self.train_acc.update(probs, y.int())
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[:2]
        logits = self(x).squeeze(-1)
        loss = self.criterion(logits, y.float())
        probs = torch.sigmoid(logits)
        self.val_acc.update(probs, y.int())
        self.val_auroc.update(probs, y.int())
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss
    def on_validation_epoch_end(self):
        self.log("val_acc", self.val_acc.compute(), prog_bar=True, sync_dist=True)
        self.log("val_auroc", self.val_auroc.compute(), prog_bar=True, sync_dist=True)
        self.val_acc.reset()
        self.val_auroc.reset()
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.learning_rate,
            weight_decay=0.05,
        )
        return optimizer