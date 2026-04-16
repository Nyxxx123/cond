"""
2D血管掩码编码器
将MIP后的2D掩码编码为条件向量
"""

import torch
import torch.nn as nn


class MaskEncoder2D(nn.Module):
    """
    2D血管掩码编码器
    输入: [B, 1, H, W]  MIP后的2D掩码
    输出: [B, cond_dim] 条件向量
    """

    def __init__(self, in_channels=1, cond_dim=256):
        super().__init__()

        # 卷积层：下采样 H,W -> H/16, W/16
        self.conv_layers = nn.Sequential(
            # Stage 1: 256->128
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Stage 2: 128->64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Stage 3: 64->32
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Stage 4: 32->16
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # 全局平均池化 + 全连接
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, cond_dim),
            nn.LayerNorm(cond_dim)
        )

    def forward(self, x):
        """
        Args:
            x: [B, 1, H, W] MIP掩码，值范围[0,1]

        Returns:
            cond: [B, cond_dim] 条件向量
        """
        # 编码
        features = self.conv_layers(x)  # [B, 256, H/16, W/16]

        # 全局池化
        pooled = self.global_pool(features)  # [B, 256, 1, 1]
        pooled = pooled.view(pooled.size(0), -1)  # [B, 256]

        # 输出条件向量
        cond = self.fc(pooled)  # [B, cond_dim]

        return cond

    def get_intermediate_features(self, x):
        """返回中间特征图，用于调试"""
        features = []
        for layer in self.conv_layers:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                features.append(x)
        return features


# 简化的编码器（更轻量）
class SimpleMaskEncoder2D(nn.Module):
    """
    更轻量的2D掩码编码器
    """

    def __init__(self, in_channels=1, cond_dim=256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, cond_dim),
            nn.LayerNorm(cond_dim)
        )

    def forward(self, x):
        return self.encoder(x)


# 测试代码
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 测试两种编码器
    mask = torch.randn(4, 1, 256, 256).to(device)

    encoder1 = MaskEncoder2D(cond_dim=256).to(device)
    encoder2 = SimpleMaskEncoder2D(cond_dim=256).to(device)

    with torch.no_grad():
        out1 = encoder1(mask)
        out2 = encoder2(mask)

    print(f"MaskEncoder2D 输出形状: {out1.shape}")
    print(f"SimpleMaskEncoder2D 输出形状: {out2.shape}")
    print(f"参数量: {sum(p.numel() for p in encoder1.parameters()):,}")