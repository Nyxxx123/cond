"""
条件注意力模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """
    交叉注意力模块
    """

    def __init__(self, embed_dim, cond_dim, num_heads=8, dropout=0.1):
        super().__init__()

        self.embed_dim = embed_dim
        self.cond_dim = cond_dim
        self.num_heads = num_heads
        self.scale = (embed_dim // num_heads) ** -0.5

        # Query投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)

        # Key/Value投影
        self.k_proj = nn.Linear(cond_dim, embed_dim)
        self.v_proj = nn.Linear(cond_dim, embed_dim)

        # 输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cond):
        """
        Args:
            x: U-Net特征图 [B, C, H, W]
            cond: 条件向量 [B, cond_dim]

        Returns:
            out: 经过交叉注意力增强的特征图 [B, C, H, W]
        """
        B, C, H, W = x.shape

        # 重塑特征图为序列形式 [B, N, C], N=H*W
        x_seq = x.flatten(2).transpose(1, 2)  # [B, N, C]

        # 计算Query
        q = self.q_proj(x_seq)  # [B, N, C]

        # 计算Key和Value（从条件向量）
        # 条件向量 [B, cond_dim] -> [B, 1, C]
        cond_expanded = cond.unsqueeze(1)  # [B, 1, cond_dim]
        k = self.k_proj(cond_expanded)  # [B, 1, C]
        v = self.v_proj(cond_expanded)  # [B, 1, C]

        # 多头注意力
        q = q.reshape(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.reshape(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.reshape(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)

        # 计算注意力权重
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # 应用注意力
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, -1, C)

        # 输出投影
        out = self.out_proj(out)  # [B, N, C]

        # 重塑回特征图
        out = out.transpose(1, 2).reshape(B, C, H, W)

        # 残差连接
        return out + x


class ConditionedBlock(nn.Module):
    """
    基础卷积块 + 时间条件 + 交叉注意力
    """

    def __init__(self, in_ch, out_ch, time_emb_dim=None, cond_dim=None, num_heads=4):
        super().__init__()

        # 时间条件投影
        self.time_mlp = None
        if time_emb_dim is not None:
            self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        # 交叉注意力（使用您的实现）
        self.cross_attn = None
        if cond_dim is not None:
            self.cross_attn = CrossAttention(
                embed_dim=out_ch,
                cond_dim=cond_dim,
                num_heads=num_heads,
                dropout=0.1
            )

        # 卷积块
        self.norm1 = nn.GroupNorm(min(4, in_ch), in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(min(4, out_ch), out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        # 跳跃连接
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb=None, cond=None):
        # 第一个卷积块
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        # 添加时间嵌入
        if self.time_mlp is not None and t_emb is not None:
            time_emb = self.time_mlp(t_emb)[:, :, None, None]
            h = h + time_emb

        # 第二个卷积块
        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)

        # 跳跃连接
        h = h + self.skip(x)

        # 交叉注意力（在残差之后）
        if self.cross_attn is not None and cond is not None:
            h = self.cross_attn(h, cond)

        return h


# 简化的加法注入版本
class AddConditionBlock(nn.Module):
    """
    使用加法注入的条件块
    """

    def __init__(self, in_ch, out_ch, time_emb_dim=None, cond_dim=None):
        super().__init__()

        self.time_mlp = None
        if time_emb_dim is not None:
            self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        # 条件投影
        self.cond_mlp = None
        if cond_dim is not None:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, out_ch),
                nn.SiLU(),
                nn.Linear(out_ch, out_ch)
            )

        self.norm1 = nn.GroupNorm(min(4, in_ch), in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.GroupNorm(min(4, out_ch), out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb=None, cond=None):
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        if self.time_mlp is not None and t_emb is not None:
            h = h + self.time_mlp(t_emb)[:, :, None, None]

        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)

        if self.cond_mlp is not None and cond is not None:
            h = h + self.cond_mlp(cond)[:, :, None, None]

        return h + self.skip(x)


def get_conditioned_block(block_type="cross_attention"):
    """选择条件块"""
    if block_type == "add":
        return AddConditionBlock
    elif block_type == "cross_attention":
        return ConditionedBlock
    else:
        raise ValueError(f"Unknown block type: {block_type}")