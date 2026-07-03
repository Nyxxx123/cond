import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    """
    正弦位置编码
    """
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

class Block(nn.Module):
    """
    基础卷积块
    """
    def __init__(self, in_ch, out_ch, time_emb_dim=None):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.time_mlp = None
        if time_emb_dim is not None:
            self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        
        # 使用GroupNorm，组数设为4
        self.norm1 = nn.GroupNorm(min(4, in_ch), in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        
        self.norm2 = nn.GroupNorm(min(4, out_ch), out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        
        # 跳跃连接
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb=None):
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
        return h + self.skip(x)

class UNet(nn.Module):
    """
    简化的U-Net模型
    """
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, time_emb_dim=256, debug=False):
        super().__init__()
        self.debug = debug
        
        # 时间步嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # 编码器
        self.enc1 = Block(in_channels, base_channels, time_emb_dim)
        self.enc2 = Block(base_channels, base_channels*2, time_emb_dim)
        self.enc3 = Block(base_channels*2, base_channels*4, time_emb_dim)
        self.enc4 = Block(base_channels*4, base_channels*8, time_emb_dim)
        
        # 瓶颈
        self.bottleneck = Block(base_channels*8, base_channels*8, time_emb_dim)
        
        # 解码器
        self.dec4 = Block(base_channels*8*2, base_channels*4, time_emb_dim)
        self.dec3 = Block(base_channels*4*2, base_channels*2, time_emb_dim)
        self.dec2 = Block(base_channels*2*2, base_channels, time_emb_dim)
        self.dec1 = Block(base_channels*2, base_channels, time_emb_dim)
        
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
        """
        调整张量的空间尺寸以匹配目标尺寸
        """
        if tensor.shape[-2:] != target_size:
            tensor = F.interpolate(tensor, size=target_size, mode='bilinear', align_corners=False)
        return tensor

    def forward(self, x, t):
        """
        x: 噪声图像 [B, C, H, W]
        t: 时间步 [B]
        """
        # 记录原始尺寸用于最终输出
        original_size = x.shape[-2:]

        # 时间嵌入
        t_emb = self.time_mlp(t)

        # 编码器
        e1 = self.enc1(x, t_emb)  # [B, 64, H, W]
        e1_size = e1.shape[-2:]

        e2 = self.enc2(self.downsample(e1), t_emb)  # [B, 128, H/2, W/2]
        e2_size = e2.shape[-2:]

        e3 = self.enc3(self.downsample(e2), t_emb)  # [B, 256, H/4, W/4]
        e3_size = e3.shape[-2:]

        e4 = self.enc4(self.downsample(e3), t_emb)  # [B, 512, H/8, W/8]
        e4_size = e4.shape[-2:]

        # 瓶颈
        b = self.bottleneck(self.downsample(e4), t_emb)  # [B, 512, H/16, W/16]

        # 解码器
        # 第4层
        b_up = self.upsample(b)  # [B, 512, H/8, W/8]
        b_up = self.adjust_size(b_up, e4_size)  # 调整到与e4相同尺寸
        d4_input = torch.cat([b_up, e4], dim=1)  # [B, 1024, H/8, W/8]
        d4 = self.dec4(d4_input, t_emb)  # [B, 256, H/8, W/8]

        # 第3层
        d4_up = self.upsample(d4)  # [B, 256, H/4, W/4]
        d4_up = self.adjust_size(d4_up, e3_size)  # 调整到与e3相同尺寸
        d3_input = torch.cat([d4_up, e3], dim=1)  # [B, 512, H/4, W/4]
        d3 = self.dec3(d3_input, t_emb)  # [B, 128, H/4, W/4]

        # 第2层
        d3_up = self.upsample(d3)  # [B, 128, H/2, W/2]
        d3_up = self.adjust_size(d3_up, e2_size)  # 调整到与e2相同尺寸
        d2_input = torch.cat([d3_up, e2], dim=1)  # [B, 256, H/2, W/2]
        d2 = self.dec2(d2_input, t_emb)  # [B, 64, H/2, W/2]

        # 第1层
        d2_up = self.upsample(d2)  # [B, 64, H, W]
        d2_up = self.adjust_size(d2_up, e1_size)  # 调整到与e1相同尺寸
        d1_input = torch.cat([d2_up, e1], dim=1)  # [B, 128, H, W]
        d1 = self.dec1(d1_input, t_emb)  # [B, 64, H, W]

        # 输出
        output = self.out_conv(d1)

        # 确保输出尺寸与原始输入一致
        if output.shape[-2:] != original_size:
            output = self.adjust_size(output, original_size)

        return output



# ==================== 测试代码 ====================
def test_unet():
    """
    测试U-Net的前向传播，验证各层输出尺寸是否正确
    """
    print("=" * 50)
    print("开始测试U-Net模型...")
    print("=" * 50)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型（启用调试模式）
    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=64,
        time_emb_dim=256,
        debug=True  # 开启调试模式查看各层尺寸
    ).to(device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 测试不同batch size
    test_batches = [1, 4]
    
    print("\n" + "=" * 50)
    print("测试不同batch size的前向传播:")
    print("=" * 50)
    
    for batch_size in test_batches:
        print(f"\n{'#'*60}")
        print(f"测试 Batch Size: {batch_size}")
        print(f"{'#'*60}")
        
        # 创建随机输入
        x = torch.randn(batch_size, 1, 28, 28).to(device)  # MNIST尺寸
        t = torch.randint(0, 1000, (batch_size,)).to(device)  # 随机时间步
        
        # 前向传播
        with torch.no_grad():
            output = model(x, t)
        
        print(f"\n最终结果:")
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {output.shape}")
        
        # 验证输出尺寸
        assert output.shape == x.shape, f"输出形状 {output.shape} 与输入形状 {x.shape} 不匹配!"
        print("✓ 输出尺寸正确")
        
        # 检查输出值范围
        print(f"输出值范围: [{output.min():.3f}, {output.max():.3f}]")
    
    print("\n" + "=" * 50)
    print("所有测试完成！")
    print("=" * 50)
    
    return model


if __name__ == "__main__":
    """
    直接运行此文件时执行测试
    """
    # 运行基本测试
    model = test_unet()