import torch
import torch.nn as nn
import lightning.pytorch as pl
from transformers import AutoModel, AutoConfig

class DeepfakeVideoMAEV2(pl.LightningModule):
    def __init__(self, learning_rate=1e-4, num_classes=1, freeze_backbone=False, distributed=False):
        super().__init__()
        self.save_hyperparameters()
        self.distributed = distributed
        model_name = "OpenGVLab/VideoMAEv2-Base" 
        print(f"🚀 Loading {model_name} (Backbone Only)...")
        
        # 1. 載入 Config (為了拿 hidden_size)
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        

        self.backbone = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            config=config
        )
        hidden_dim = config.model_config['embed_dim'] if isinstance(config.model_config, dict) else config.model_config.embed_dim
        # 3. 手動建立分類頭 (Head)
        # VideoMAE V2 Base 的 hidden_size 通常是 768
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # === 凍結邏輯 (Partial Fine-tuning) ===
        if freeze_backbone:
            print("❄️  Freezing Backbone... Only training the Classifier!")
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.classifier.parameters():
                param.requires_grad = True
        else:
            # 建議至少凍結 Patch Embedding (這在小數據集上很有效)
            print("🔧 Full Fine-tuning (with frozen patch_embed)")
            for name, param in self.backbone.named_parameters():
                 if 'patch_embed' in name:
                     param.requires_grad = False

    def forward(self, x):
        # 1. 維度修正 (Input Shape Fix)
        # 確保輸入是 (B, C, T, H, W)
        if x.shape[1] != 3 and x.shape[2] == 3:
             x = x.permute(0, 2, 1, 3, 4)
        
        # 2. 通過 Backbone
        outputs = self.backbone(x)
        
        # 3. 輸出處理 (Output Handling)
        if isinstance(outputs, torch.Tensor):
            features = outputs
        elif hasattr(outputs, 'last_hidden_state'):
            features = outputs.last_hidden_state
        else:
            features = outputs[0]
            
        # === 4. 智慧池化 (Smart Pooling) [關鍵修正] ===
        # 檢查維度：
        # 如果是 (Batch, Seq, Hidden) -> 需要 Pooling
        # 如果是 (Batch, Hidden)      -> 已經 Pool 過了，直接用
        
        if features.dim() == 3:
            pooled_features = features.mean(dim=1)
        elif features.dim() == 2:
            pooled_features = features
        else:
            raise ValueError(f"Unexpected features shape: {features.shape}, expected 2D or 3D tensor.")
        
        # 5. 通過分類頭
        logits = self.classifier(pooled_features)
        
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch[:2]
        logits = self(x)
        loss = nn.BCEWithLogitsLoss()(logits.squeeze(), y.float())
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[:2]
        logits = self(x)
        loss = nn.BCEWithLogitsLoss()(logits.squeeze(), y.float())
        preds = torch.sigmoid(logits.squeeze()) > 0.5
        acc = (preds == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()), 
            lr=self.hparams.learning_rate, 
            weight_decay=0.05
        )