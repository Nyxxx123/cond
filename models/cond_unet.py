"""
条件U-Net
接收噪声图像和血管掩码条件，生成肺动脉造影图像
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.encoder import MaskEncoder2D, MaskEncoder3D  # 修改：从encoder导入
from models.cond_attention import ConditionedBlock, AddConditionBlock


class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConditionalUNet(nn.Module):
    """
    条件U-Net

    输入:
        x: 噪声图像 [B, 1, H, W]
        mask: 血管掩码（2D: [B,1,H,W] 或 3D: [B,1,D,H,W]）
        t: 时间步 [B]

    输出:
        预测噪声 [B, 1, H, W]
    """

    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 base_channels=64,
                 cond_dim=256,
                 time_emb_dim=256,
                 block_type="cross_attention",
                 mask_type="3d"):  # 新增参数： "2d" 或 "3d"
        super().__init__()

        self.base_channels = base_channels
        self.cond_dim = cond_dim
        self.mask_type = mask_type

        # 选择条件块类型
        if block_type == "add":
            BlockClass = AddConditionBlock
        else:
            BlockClass = ConditionedBlock

        # 根据类型选择编码器（修改：动态选择）
        if mask_type == "2d":
            self.mask_encoder = MaskEncoder2D(in_channels=1, cond_dim=cond_dim)
        else:
            self.mask_encoder = MaskEncoder3D(in_channels=1, cond_dim=cond_dim)

        # 时间步嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # 初始卷积
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # 编码器
        self.enc1 = BlockClass(base_channels, base_channels, time_emb_dim, cond_dim)
        self.enc2 = BlockClass(base_channels, base_channels * 2, time_emb_dim, cond_dim)
        self.enc3 = BlockClass(base_channels * 2, base_channels * 4, time_emb_dim, cond_dim)
        self.enc4 = BlockClass(base_channels * 4, base_channels * 8, time_emb_dim, cond_dim)

        # 瓶颈
        self.bottleneck = BlockClass(base_channels * 8, base_channels * 8, time_emb_dim, cond_dim)

        # 解码器
        self.dec4 = BlockClass(base_channels * 8 * 2, base_channels * 4, time_emb_dim, cond_dim)
        self.dec3 = BlockClass(base_channels * 4 * 2, base_channels * 2, time_emb_dim, cond_dim)
        self.dec2 = BlockClass(base_channels * 2 * 2, base_channels, time_emb_dim, cond_dim)
        self.dec1 = BlockClass(base_channels * 2, base_channels, time_emb_dim, cond_dim)

        # 下采样/上采样
        self.downsample = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # 输出层
        self.out_conv = nn.Sequential(
            nn.GroupNorm(min(4, base_channels), base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, kernel_size=1)
        )

    def adjust_size(self, tensor, target_size):
        """调整张量尺寸"""
        if tensor.shape[-2:] != target_size:
            tensor = F.interpolate(tensor, size=target_size, mode='bilinear', align_corners=False)
        return tensor

    def forward(self, x, mask, t):
        """
        Args:
            x: 噪声图像 [B, 1, H, W]
            mask: 血管掩码（2D或3D）
            t: 时间步 [B]
        """
        original_size = x.shape[-2:]

        # 编码条件
        cond = self.mask_encoder(mask)  # [B, cond_dim]

        # 时间嵌入
        t_emb = self.time_mlp(t)

        # 初始卷积
        h = self.init_conv(x)

        # 编码器
        e1 = self.enc1(h, t_emb, cond)
        e1_size = e1.shape[-2:]

        e2 = self.enc2(self.downsample(e1), t_emb, cond)
        e2_size = e2.shape[-2:]

        e3 = self.enc3(self.downsample(e2), t_emb, cond)
        e3_size = e3.shape[-2:]

        e4 = self.enc4(self.downsample(e3), t_emb, cond)
        e4_size = e4.shape[-2:]

        # 瓶颈
        b = self.bottleneck(self.downsample(e4), t_emb, cond)

        # 解码器
        b_up = self.upsample(b)
        b_up = self.adjust_size(b_up, e4_size)
        d4 = self.dec4(torch.cat([b_up, e4], dim=1), t_emb, cond)

        d4_up = self.upsample(d4)
        d4_up = self.adjust_size(d4_up, e3_size)
        d3 = self.dec3(torch.cat([d4_up, e3], dim=1), t_emb, cond)

        d3_up = self.upsample(d3)
        d3_up = self.adjust_size(d3_up, e2_size)
        d2 = self.dec2(torch.cat([d3_up, e2], dim=1), t_emb, cond)

        d2_up = self.upsample(d2)
        d2_up = self.adjust_size(d2_up, e1_size)
        d1 = self.dec1(torch.cat([d2_up, e1], dim=1), t_emb, cond)

        # 输出
        output = self.out_conv(d1)

        # 确保尺寸一致
        if output.shape[-2:] != original_size:
            output = self.adjust_size(output, original_size)

        return output