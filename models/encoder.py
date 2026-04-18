"""
编码器模块：支持2D和3D掩码编码
"""

import torch
import torch.nn as nn


# ==================== 2D编码器 ====================

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
        features = self.conv_layers(x)  # [B, 256, H/16, W/16]
        pooled = self.global_pool(features)  # [B, 256, 1, 1]
        pooled = pooled.view(pooled.size(0), -1)  # [B, 256]
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


# ==================== 3D编码器 ====================

class MaskEncoder3D(nn.Module):
    """
    3D血管掩码编码器
    输入: [B, 1, D, H, W] 3D血管掩码
    输出: [B, cond_dim] 条件向量
    """

    def __init__(self, in_channels=1, cond_dim=256):
        super().__init__()

        # 3D卷积层：逐步下采样
        self.conv_layers = nn.Sequential(
            # Stage 1: 64->32 (假设输入64)
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            # Stage 2: 32->16
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            # Stage 3: 16->8
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            # Stage 4: 8->4
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        )

        # 全局平均池化 + 全连接
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, cond_dim),
            nn.LayerNorm(cond_dim)
        )

    def forward(self, x):
        """
        Args:
            x: [B, 1, D, H, W] 3D掩码，值范围[0,1]

        Returns:
            cond: [B, cond_dim] 条件向量
        """
        features = self.conv_layers(x)  # [B, 256, 4, 4, 4]
        pooled = self.global_pool(features)  # [B, 256, 1, 1, 1]
        pooled = pooled.view(pooled.size(0), -1)  # [B, 256]
        cond = self.fc(pooled)  # [B, cond_dim]
        return cond

    def get_intermediate_features(self, x):
        """返回中间特征图，用于调试"""
        features = []
        for layer in self.conv_layers:
            x = layer(x)
            if isinstance(layer, nn.Conv3d):
                features.append(x)
        return features


class SimpleMaskEncoder3D(nn.Module):
    """
    更轻量的3D掩码编码器
    """

    def __init__(self, in_channels=1, cond_dim=256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(256, cond_dim),
            nn.LayerNorm(cond_dim)
        )

    def forward(self, x):
        return self.encoder(x)


# ==================== 多模态编码器（扩展用） ====================

class MultiModalEncoder(nn.Module):
    """
    多模态编码器：支持多种条件融合
    """

    def __init__(self, cond_dim=256, use_2d_mask=False, use_3d_mask=True, use_angle=False):
        super().__init__()

        self.use_2d_mask = use_2d_mask
        self.use_3d_mask = use_3d_mask
        self.use_angle = use_angle

        total_dim = 0

        if use_2d_mask:
            self.mask_2d_encoder = SimpleMaskEncoder2D(cond_dim=cond_dim)
            total_dim += cond_dim

        if use_3d_mask:
            self.mask_3d_encoder = SimpleMaskEncoder3D(cond_dim=cond_dim)
            total_dim += cond_dim

        if use_angle:
            self.angle_encoder = AngleEncoder(cond_dim=cond_dim)
            total_dim += cond_dim

        # 融合层
        if total_dim > cond_dim:
            self.fusion = nn.Sequential(
                nn.Linear(total_dim, cond_dim * 2),
                nn.ReLU(inplace=True),
                nn.Linear(cond_dim * 2, cond_dim),
                nn.LayerNorm(cond_dim)
            )
        else:
            self.fusion = nn.Identity()

        self.cond_dim = cond_dim

    def forward(self, mask_2d=None, mask_3d=None, angle=None):
        features = []

        if self.use_2d_mask and mask_2d is not None:
            features.append(self.mask_2d_encoder(mask_2d))

        if self.use_3d_mask and mask_3d is not None:
            features.append(self.mask_3d_encoder(mask_3d))

        if self.use_angle and angle is not None:
            features.append(self.angle_encoder(angle))

        if len(features) == 0:
            return torch.zeros(1, self.cond_dim)

        if len(features) == 1:
            return features[0]

        concat = torch.cat(features, dim=1)
        return self.fusion(concat)


class AngleEncoder(nn.Module):
    """
    角度编码器（参考您的另一个项目）
    输入: [B, 9] 旋转矩阵
    输出: [B, cond_dim]
    """

    def __init__(self, in_dim=9, cond_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, cond_dim),
            nn.LayerNorm(cond_dim)
        )

    def forward(self, x):
        return self.mlp(x)


# ==================== 测试代码 ====================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 50)
    print("测试编码器")
    print("=" * 50)

    # 测试2D编码器
    print("\n1. 测试2D编码器:")
    mask_2d = torch.randn(4, 1, 256, 256).to(device)
    encoder_2d = MaskEncoder2D(cond_dim=256).to(device)
    encoder_2d_simple = SimpleMaskEncoder2D(cond_dim=256).to(device)

    with torch.no_grad():
        out1 = encoder_2d(mask_2d)
        out2 = encoder_2d_simple(mask_2d)

    print(f"  MaskEncoder2D 输出: {out1.shape}")
    print(f"  SimpleMaskEncoder2D 输出: {out2.shape}")
    print(f"  参数量: {sum(p.numel() for p in encoder_2d.parameters()):,}")

    # 测试3D编码器
    print("\n2. 测试3D编码器:")
    mask_3d = torch.randn(4, 1, 64, 64, 64).to(device)
    encoder_3d = MaskEncoder3D(cond_dim=256).to(device)
    encoder_3d_simple = SimpleMaskEncoder3D(cond_dim=256).to(device)

    with torch.no_grad():
        out3 = encoder_3d(mask_3d)
        out4 = encoder_3d_simple(mask_3d)

    print(f"  MaskEncoder3D 输出: {out3.shape}")
    print(f"  SimpleMaskEncoder3D 输出: {out4.shape}")
    print(f"  参数量: {sum(p.numel() for p in encoder_3d.parameters()):,}")

    # 测试多模态编码器
    print("\n3. 测试多模态编码器:")
    multi_encoder = MultiModalEncoder(
        cond_dim=256,
        use_2d_mask=True,
        use_3d_mask=True,
        use_angle=True
    ).to(device)

    angle = torch.randn(4, 9).to(device)
    with torch.no_grad():
        out5 = multi_encoder(mask_2d=mask_2d, mask_3d=mask_3d, angle=angle)

    print(f"  MultiModalEncoder 输出: {out5.shape}")
    print(f"  参数量: {sum(p.numel() for p in multi_encoder.parameters()):,}")

    print("\n" + "=" * 50)
    print("测试完成！")
    print("=" * 50)