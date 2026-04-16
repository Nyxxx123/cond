"""
扩散过程核心实现
支持 DDPM 和 DDIM 两种采样方式
支持条件生成
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


class GaussianDiffusion:
    """
    高斯扩散过程
    支持 DDPM 和 DDIM 采样
    支持条件生成（传入 mask 条件）
    """
    def __init__(self, betas, device):
        """
        betas: 噪声调度序列
        device: 计算设备
        """
        self.betas = betas.to(device)
        self.device = device
        self.timesteps = len(betas)

        # 计算相关参数
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # 确保 alphas_cumprod 在正确的设备上
        alphas_cumprod = alphas_cumprod.to(device)

        # 计算 alphas_cumprod_prev（用于 DDIM）
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]])

        # 注册为属性
        self.alphas = alphas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.alphas_cumprod_prev = alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)

    def _extract(self, a, t, x_shape):
        """
        从张量a中提取时间步t对应的值
        """
        a = a.to(self.device)
        batch_size = t.shape[0]

        t_cpu = t.cpu()
        out = a.cpu().gather(-1, t_cpu).to(self.device)

        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise=None):
        """
        前向扩散：从x0生成xt
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, model, x_start, t, cond=None, noise=None):
        """
        计算训练损失（支持条件）

        Args:
            model: 条件UNet模型
            x_start: 原始图像 [B, C, H, W]
            t: 时间步 [B]
            cond: 条件（如MIP掩码）[B, 1, H, W]
            noise: 噪声（可选）
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start, t, noise)

        # 条件前向传播
        if cond is not None:
            predicted_noise = model(x_noisy, cond, t)
        else:
            predicted_noise = model(x_noisy, t)

        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index, cond=None):
        """
        单步逆向采样（支持条件）
        """
        t = t.long().to(self.device)

        # 提取参数
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = 1.0 / torch.sqrt(self._extract(self.alphas, t, x.shape))

        # 预测噪声（条件）
        if cond is not None:
            predicted_noise = model(x, cond, t)
        else:
            predicted_noise = model(x, t)

        # 计算均值
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        # 如果不是最后一步，添加噪声
        if t_index == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            posterior_variance = betas_t
            return model_mean + torch.sqrt(posterior_variance) * noise

    @torch.no_grad()
    def sample_ddpm(self, model, image_size, batch_size=16, channels=1,
                    cond=None, progress=True):
        """
        完整DDPM采样循环（支持条件）

        Args:
            model: 条件UNet
            image_size: 图像尺寸
            batch_size: 批次大小
            channels: 通道数
            cond: 条件（如MIP掩码）[B, 1, H, W] 或 None
            progress: 是否显示进度条
        """
        shape = (batch_size, channels, image_size, image_size)
        img = torch.randn(shape, device=self.device)

        # 如果条件存在但batch不匹配，则复制
        if cond is not None and cond.shape[0] != batch_size:
            cond = cond.repeat(batch_size, 1, 1, 1)

        intermediates = []
        indices = list(range(self.timesteps))[::-1]

        if progress:
            indices = tqdm(indices, desc="DDPM Sampling")

        for i in indices:
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            img = self.p_sample(model, img, t, i, cond=cond)

            if i % 100 == 0 or i == self.timesteps - 1 or i == 0:
                intermediates.append(img.cpu())

        return img, intermediates

    @torch.no_grad()
    def sample_timestep_ddim(self, model, x, t, cond=None, eta=0.0):
        """
        DDIM 单步采样（支持条件）
        """
        # 获取当前时间步的参数
        alpha_cumprod_t = self._extract(self.alphas_cumprod, t, x.shape)
        alpha_cumprod_t_prev = self._extract(self.alphas_cumprod_prev, t, x.shape)

        sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
        sqrt_alpha_cumprod_t_prev = torch.sqrt(alpha_cumprod_t_prev)
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alpha_cumprod_t)

        # 预测噪声（条件）
        if cond is not None:
            eps_theta = model(x, cond, t)
        else:
            eps_theta = model(x, t)

        # 预测 x0
        x0_pred = (x - sqrt_one_minus_alpha_cumprod_t * eps_theta) / sqrt_alpha_cumprod_t
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        # 计算 sigma
        sigma = eta * torch.sqrt(
            (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) *
            (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
        )

        # 生成噪声
        noise = torch.randn_like(x) if eta > 0 else 0

        # 计算方向
        dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigma**2) * eps_theta

        # 更新
        x_prev = sqrt_alpha_cumprod_t_prev * x0_pred + dir_xt + sigma * noise

        return x_prev

    @torch.no_grad()
    def sample_ddim(self, model, image_size, batch_size=16, channels=1,
                    ddim_steps=50, eta=0.0, cond=None, progress=True):
        """
        DDIM 快速采样（支持条件）
        """
        shape = (batch_size, channels, image_size, image_size)
        img = torch.randn(shape, device=self.device)

        # 如果条件存在但batch不匹配，则复制
        if cond is not None and cond.shape[0] != batch_size:
            cond = cond.repeat(batch_size, 1, 1, 1)

        intermediates = []

        # 生成时间步序列
        ddim_timesteps = np.linspace(0, self.timesteps - 1, ddim_steps, dtype=int)
        ddim_timesteps = ddim_timesteps[::-1]

        if progress:
            pbar = tqdm(range(len(ddim_timesteps)), desc="DDIM Sampling")

        for i, step in enumerate(ddim_timesteps):
            t = torch.full((batch_size,), step, device=self.device, dtype=torch.long)
            img = self.sample_timestep_ddim(model, img, t, cond=cond, eta=eta)
            img = torch.clamp(img, -1.0, 1.0)

            if i % max(1, len(ddim_timesteps) // 10) == 0 or i == len(ddim_timesteps) - 1:
                intermediates.append(img.cpu())

            if progress:
                pbar.update(1)

        if progress:
            pbar.close()

        return img, intermediates

    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=1,
               sampler_type="ddpm", ddim_steps=50, eta=0.0,
               cond=None, progress=True):
        """
        统一的采样接口（支持条件）

        Args:
            model: 条件UNet
            image_size: 图像尺寸
            batch_size: 批次大小
            channels: 通道数
            sampler_type: "ddpm" 或 "ddim"
            ddim_steps: DDIM采样步数
            eta: DDIM随机性
            cond: 条件（如MIP掩码）[B, 1, H, W] 或 None
            progress: 是否显示进度条
        """
        if sampler_type == "ddpm":
            return self.sample_ddpm(model, image_size, batch_size, channels,
                                    cond=cond, progress=progress)
        elif sampler_type == "ddim":
            return self.sample_ddim(model, image_size, batch_size, channels,
                                    ddim_steps, eta, cond=cond, progress=progress)
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")


# 测试函数
def test_diffusion():
    """
    测试条件扩散过程
    """
    from models.cond_unet import ConditionalUNet

    print("=" * 50)
    print("测试条件扩散过程（DDPM + DDIM）...")
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建简单的噪声调度
    timesteps = 100
    betas = torch.linspace(0.0001, 0.02, timesteps)

    # 创建扩散过程
    diffusion = GaussianDiffusion(betas, device)

    # 创建条件模型
    model = ConditionalUNet(
        in_channels=1,
        out_channels=1,
        base_channels=32,
        cond_dim=256,
        block_type="cross_attention",
    ).to(device)
    model.eval()

    # 测试前向扩散（带条件）
    print("\n测试前向扩散（带条件）:")
    x = torch.randn(2, 1, 28, 28).to(device)
    mask = torch.rand(2, 1, 28, 28).to(device)
    t = torch.randint(0, timesteps, (2,)).to(device)

    x_noisy = diffusion.q_sample(x, t)
    print(f"原始图像形状: {x.shape}")
    print(f"条件掩码形状: {mask.shape}")
    print(f"加噪后图像形状: {x_noisy.shape}")

    # 测试损失计算
    loss = diffusion.p_losses(model, x, t, cond=mask)
    print(f"条件损失: {loss.item():.4f}")

    # 测试 DDPM 条件采样
    print("\n测试 DDPM 条件采样（100步）:")
    try:
        samples_ddpm, _ = diffusion.sample(
            model,
            image_size=28,
            batch_size=2,
            channels=1,
            sampler_type="ddpm",
            cond=mask,
            progress=False
        )
        print(f"✓ DDPM 生成样本形状: {samples_ddpm.shape}")
    except Exception as e:
        print(f"✗ DDPM 采样失败: {e}")

    # 测试 DDIM 条件采样
    print("\n测试 DDIM 条件采样（20步）:")
    try:
        samples_ddim, _ = diffusion.sample(
            model,
            image_size=28,
            batch_size=2,
            channels=1,
            sampler_type="ddim",
            ddim_steps=20,
            eta=0.0,
            cond=mask,
            progress=False
        )
        print(f"✓ DDIM 生成样本形状: {samples_ddim.shape}")
    except Exception as e:
        print(f"✗ DDIM 采样失败: {e}")

    print("\n" + "=" * 50)
    print("测试完成！")
    print("=" * 50)


if __name__ == "__main__":
    test_diffusion()