# modeling_cu_classifier.py

from transformers import DistilBertPreTrainedModel, DistilBertModel
import torch
import torch.nn as nn

class CUClassifierHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.downsample = nn.Linear(config.dim_embedding, config.dim_embedding)

        self.fc1 = nn.Sequential(
            nn.Linear(config.dim_embedding, config.hidden_layers),
            nn.GELU(),
            nn.LayerNorm(config.hidden_layers),
            nn.Dropout(0.30),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(config.hidden_layers, config.hidden_layers),
            nn.GELU(),
            nn.LayerNorm(config.hidden_layers),
            nn.Dropout(0.55),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(config.hidden_layers, config.dim_embedding),
            nn.GELU(),
            nn.LayerNorm(config.dim_embedding),
            nn.Dropout(0.30),
        )

        self.out = nn.Linear(config.dim_embedding, config.num_labels)

    def forward(self, x):
        identity = x.clone()
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.downsample(identity) + x
        return self.out(x)

class DistilBertForCUClassification(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        config.dim_embedding = config.hidden_size
        config.hidden_layers = getattr(config, "hidden_layers", 512)
        self.classifier = CUClassifierHead(config)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state[:, 0]  # [CLS] token embedding
        logits = self.classifier(x)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
