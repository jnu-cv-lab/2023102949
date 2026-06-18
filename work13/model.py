import torch
import torch.nn as nn


class SkeletonTransformer(nn.Module):
    """
    输入:  [B, T, 132]
    输出:  [B, num_classes]
    """

    def __init__(
        self,
        input_dim=132,
        target_frames=30,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        num_classes=6,
        dropout=0.1,
    ):
        super().__init__()

        self.target_frames = target_frames
        self.d_model = d_model

        # 把每一帧 132 维骨架特征映射到 Transformer 的 d_model 维
        self.input_projection = nn.Linear(input_dim, d_model)

        # 可学习的位置编码，表示第 1 帧、第 2 帧……的时间位置信息
        self.position_embedding = nn.Parameter(
            torch.randn(1, target_frames, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x):
        """
        x shape: [B, T, 132]
        """
        x = self.input_projection(x)

        # 加上位置编码
        x = x + self.position_embedding[:, :x.size(1), :]

        x = self.transformer_encoder(x)

        # Mean Pooling：对所有时间帧取平均
        x = x.mean(dim=1)

        logits = self.classifier(x)

        return logits