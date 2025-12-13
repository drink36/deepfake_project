import torch
import torch.nn as nn
import lightning.pytorch as pl
from transformers import AutoModel, AutoConfig


class DeepfakeVideoMAEV2(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 1e-5,
        num_classes: int = 1,
        freeze_backbone: bool = False,
        distributed: bool = False,
        num_frames: int = 16,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.distributed = distributed
        model_name = "OpenGVLab/VideoMAEv2-Base"
        print(f"🚀 Loading backbone: {model_name}")
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        self.backbone = AutoModel.from_pretrained(
            model_name,
            config=config,
            trust_remote_code=True,
        )

        hidden_dim = None
        if hasattr(config, "model_config"):
            mc = config.model_config
            hidden_dim = mc.get("embed_dim", None)
        if hasattr(config, "num_frames"):
            config.num_frames = num_frames

        hidden_dim = (
            config.model_config["embed_dim"]
            if isinstance(config.model_config, dict)
            else config.model_config.embed_dim
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
        self.register_buffer("img_mean", mean, persistent=False)
        self.register_buffer("img_std", std, persistent=False)

        if freeze_backbone:
            print("❄️  Freezing backbone (only classifier trainable)")
            for p in self.backbone.parameters():
                p.requires_grad = False
        else:
            print("🔧 Full fine-tuning backbone (except patch_embed)")
            for p in self.backbone.parameters():
                p.requires_grad = True
            # 可選：鎖 patch_embed 穩定一點
            for name, p in self.backbone.named_parameters():
                if "patch_embed" in name:
                    p.requires_grad = False

        # 6. loss
        self.criterion = nn.BCEWithLogitsLoss()
    def ensure_train_mode(self):
        """Make sure every layer in backbone enters train() mode."""
        for m in self.backbone.modules():
            m.train()
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.img_mean) / self.img_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"[VideoMAE] Expected 5D video tensor, got {x.shape}")

        if x.shape[1] != 3 and x.shape[2] == 3:

            x = x.permute(0, 2, 1, 3, 4)


        x = self._normalize(x)

        # backbone forward
        outputs = self.backbone(x)

        # VideoMAEv2_Base 根據 config:
        # - num_classes = 0, use_mean_pooling = True
        #   很有機會直接回傳 (B, hidden_dim) tensor 作為 pooled feature
        if isinstance(outputs, torch.Tensor):
            features = outputs                      # 可能是 (B, hidden_dim) 或 (B, L, hidden_dim)
        elif hasattr(outputs, "last_hidden_state"):
            # 標準 HuggingFace 型式
            features = outputs.last_hidden_state    # (B, L, hidden_dim)
        else:
            # 一般情況下 outputs[0] 是 last_hidden_state
            features = outputs[0]

        # 根據 shape 做處理：
        if features.dim() == 2:
            # (B, hidden_dim) → 已經 pool 好了，直接用
            pooled = features
        elif features.dim() == 3:
            # (B, L, hidden_dim) → 再 mean pool 一次（保險）
            pooled = features.mean(dim=1)
        else:
            raise ValueError(f"[VideoMAE] Unexpected backbone output shape: {features.shape}")

        logits = self.classifier(pooled)            # (B, 1)
        return logits

    # ---------- Lightning hooks ----------
    def training_step(self, batch, batch_idx):
        self.ensure_train_mode()
        if batch_idx == 0:
            print("Backbone TRAIN modules:", sum(1 for m in self.backbone.modules() if m.training), "Backbone EVAL modules:", sum(1 for m in self.backbone.modules() if not m.training))
        x, y = batch[:2]                # x: (B, 3, T, H, W), y: (B,)
        logits = self(x).squeeze(-1)    # (B,)
        loss = self.criterion(logits, y.float())

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[:2]
        logits = self(x).squeeze(-1)
        loss = self.criterion(logits, y.float())
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        acc = (preds == y).float().mean()

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)
        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.learning_rate,
            weight_decay=0.05,
        )
        return optimizer
