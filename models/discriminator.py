"""
判别器网络 - 用于对抗训练
基于 DCGAN 架构的 PatchGAN 判别器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    """
    PatchGAN 判别器
    输出是一个特征图（每个patch的真实性判断），而不是单一标量
    这种设计对小细节更敏感，训练也更稳定
    """

    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()

        # 卷积层序列
        self.conv_layers = nn.Sequential(
            # 输入: [B, 1, H, W]
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # 输出层：1通道，表示每个patch的真实性
            nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x):
        """
        Args:
            x: 输入图像 [B, C, H, W]
        Returns:
            logits: [B, 1, H/16, W/16] 每个patch的真实性得分
        """
        return self.conv_layers(x)


class NLayerDiscriminator(nn.Module):
    """
    更灵活的判别器，可配置层数
    参考 pix2pixHD 的多尺度判别器设计
    """

    def __init__(self, in_channels=1, base_channels=64, n_layers=4, norm_layer=nn.BatchNorm2d):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        # 逐层增加通道数
        for i in range(1, n_layers):
            mult = min(2 ** i, 8)
            in_ch = base_channels * min(2 ** (i - 1), 8)
            out_ch = base_channels * mult

            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                norm_layer(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        # 最后一层
        mult = min(2 ** (n_layers - 1), 8)
        in_ch = base_channels * mult
        layers += [
            nn.Conv2d(in_ch, 1, kernel_size=4, stride=1, padding=1)
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ==================== 新增：条件判别器 ====================
class CondDiscriminator(nn.Module):
    """
    条件 PatchGAN 判别器
    输入: 图像 [B, C, H, W] 和 条件向量 [B, cond_dim]
    输出: [B, 1, H/16, W/16] 的 logits
    """
    def __init__(self, in_channels=1, cond_dim=256, base_channels=64):
        super().__init__()

        # 条件向量投影到与特征图通道数匹配的空间
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 卷积层：输出特征图
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 最后一层：将条件投影加到特征图上再输出
        self.out_conv = nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x, cond):
        """
        Args:
            x: 输入图像 [B, 1, H, W]
            cond: 条件向量 [B, cond_dim]
        Returns:
            logits: [B, 1, H/16, W/16]
        """
        # 提取图像特征
        features = self.conv_layers(x)          # [B, 512, 16, 16]
        B, C, H, W = features.shape

        # 投影条件向量并加到特征图上（广播）
        cond_embed = self.cond_proj(cond)       # [B, 512]
        cond_embed = cond_embed.view(B, C, 1, 1)  # [B, 512, 1, 1]
        features = features + cond_embed

        # 输出判别分数
        out = self.out_conv(features)           # [B, 1, 16, 16]
        return out
# =====================================================


# 测试代码
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 测试判别器
    disc = Discriminator(in_channels=1).to(device)
    x = torch.randn(4, 1, 256, 256).to(device)
    out = disc(x)
    print(f"Discriminator output shape: {out.shape}")  # [4, 1, 16, 16]
    print(f"参数量: {sum(p.numel() for p in disc.parameters()):,}")

    # 测试条件判别器
    cond_disc = CondDiscriminator(in_channels=1, cond_dim=256).to(device)
    cond = torch.randn(4, 256).to(device)
    out_cond = cond_disc(x, cond)
    print(f"CondDiscriminator output shape: {out_cond.shape}")