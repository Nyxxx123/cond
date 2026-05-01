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
            # Stage 1: 64->32
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


# ==================== 多模态编码器（扩展用） ====================

class AngleEncoder(nn.Module):
    """
    角度编码器
    输入: [B, angle_dim] 角度表示（四元数/旋转矩阵/欧拉角）
    输出: [B, cond_dim] 角度特征向量
    """

    def __init__(self, angle_dim=4, cond_dim=256):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(angle_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, cond_dim),
            nn.LayerNorm(cond_dim)
        )

    def forward(self, angle):
        """
        Args:
            angle: [B, angle_dim] 角度表示（四元数/旋转矩阵/欧拉角）
        Returns:
            angle_feat: [B, cond_dim] 角度特征
        """
        return self.mlp(angle)


# ==================== 新增：无造影CT编码器 ====================
class XRayEncoder(nn.Module):
    """
    2D 无造影CT图像编码器
    输入: [B, 1, H, W]
    输出: [B, cond_dim] 条件向量
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


class MultiConditionEncoder(nn.Module):
    """
    多条件编码器：融合掩码和角度信息
    （扩展支持无造影CT）
    """

    def __init__(self,
                 mask_type="3d",
                 mask_cond_dim=256,
                 angle_cond_dim=256,
                 use_angle=True,
                 angle_dim=4,
                 use_non_angio=False,           # 新增参数
                 non_angio_cond_dim=256,        # 新增参数
                 fused_cond_dim=256):
        super().__init__()

        self.use_angle = use_angle
        self.use_non_angio = use_non_angio      # 新增

        # 掩码编码器
        if mask_type == "2d":
            self.mask_encoder = MaskEncoder2D(cond_dim=mask_cond_dim)
        else:
            self.mask_encoder = MaskEncoder3D(cond_dim=mask_cond_dim)

        # 角度编码器
        if use_angle:
            self.angle_encoder = AngleEncoder(angle_dim=angle_dim, cond_dim=angle_cond_dim)

        # 新增：无造影CT编码器
        if use_non_angio:
            self.non_angio_encoder = XRayEncoder(in_channels=1, cond_dim=non_angio_cond_dim)

        # 特征融合层
        total_dim = mask_cond_dim
        if use_angle:
            total_dim += angle_cond_dim
        if use_non_angio:                     # 新增
            total_dim += non_angio_cond_dim

        if total_dim > fused_cond_dim:
            self.fusion = nn.Sequential(
                nn.Linear(total_dim, fused_cond_dim * 2),
                nn.ReLU(inplace=True),
                nn.Linear(fused_cond_dim * 2, fused_cond_dim),
                nn.LayerNorm(fused_cond_dim)
            )
        else:
            self.fusion = nn.Identity()

        self.fused_cond_dim = fused_cond_dim

    def forward(self, mask, angle=None, non_angio=None):   # 新增 non_angio 参数
        """
        Args:
            mask: [B, 1, D, H, W] 或 [B, 1, H, W]
            angle: [B, angle_dim] 四元数/旋转矩阵/欧拉角，可选
            non_angio: [B, 1, H, W] 无造影CT图像，可选

        Returns:
            cond: [B, fused_cond_dim] 融合后的条件向量
        """
        # 编码掩码
        mask_feat = self.mask_encoder(mask)  # [B, mask_cond_dim]
        features = [mask_feat]

        # 编码角度
        if self.use_angle and angle is not None:
            angle_feat = self.angle_encoder(angle)  # [B, angle_cond_dim]
            features.append(angle_feat)

        # 新增：编码无造影CT
        if self.use_non_angio and non_angio is not None:
            non_angio_feat = self.non_angio_encoder(non_angio)  # [B, non_angio_cond_dim]
            features.append(non_angio_feat)

        # 融合
        if len(features) == 1:
            cond = features[0]
        else:
            concat = torch.cat(features, dim=1)
            cond = self.fusion(concat)

        return cond