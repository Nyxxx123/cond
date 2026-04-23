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


# 测试代码
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 测试判别器
    disc = Discriminator(in_channels=1).to(device)
    x = torch.randn(4, 1, 256, 256).to(device)
    out = disc(x)
    print(f"Discriminator output shape: {out.shape}")  # [4, 1, 16, 16]
    print(f"参数量: {sum(p.numel() for p in disc.parameters()):,}")