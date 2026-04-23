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

    def p_losses(self, model, x_start, t, mask=None, angle=None, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start, t, noise)

        # 传入多条件
        predicted_noise = model(x_noisy, mask, angle=angle, t=t)

        return F.mse_loss(predicted_noise, noise)

    # 修改 p_sample 方法
    @torch.no_grad()
    def p_sample(self, model, x, t, t_index, mask=None, angle=None):
        t = t.long().to(self.device)

        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = 1.0 / torch.sqrt(self._extract(self.alphas, t, x.shape))

        predicted_noise = model(x, mask, angle=angle, t=t)

        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            posterior_variance = betas_t
            return model_mean + torch.sqrt(posterior_variance) * noise

    @torch.no_grad()
    def sample_ddpm(self, model, image_size, batch_size=16, channels=1,
                    mask=None, angle=None, progress=True):
        """
        完整DDPM采样循环（支持条件）

        Args:
            model: 条件UNet
            image_size: 图像尺寸
            batch_size: 批次大小
            channels: 通道数
            mask: 掩码条件（2D或3D）
            angle: 角度条件（四元数）
            progress: 是否显示进度条
        """
        shape = (batch_size, channels, image_size, image_size)
        img = torch.randn(shape, device=self.device)

        # 如果条件存在但batch不匹配，则复制
        if mask is not None and mask.shape[0] != batch_size:
            if mask.dim() == 4:
                mask = mask.repeat(batch_size, 1, 1, 1)
            elif mask.dim() == 5:
                mask = mask.repeat(batch_size, 1, 1, 1, 1)

        if angle is not None and angle.shape[0] != batch_size:
            angle = angle.repeat(batch_size, 1)

        intermediates = []
        indices = list(range(self.timesteps))[::-1]

        if progress:
            indices = tqdm(indices, desc="DDPM Sampling")

        for i in indices:
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            # 修改这里：传入 mask 和 angle，不再用 cond
            img = self.p_sample(model, img, t, i, mask=mask, angle=angle)

            if i % 100 == 0 or i == self.timesteps - 1 or i == 0:
                intermediates.append(img.cpu())

        return img, intermediates

    @torch.no_grad()
    def sample_timestep_ddim(self, model, x, t, mask=None, angle=None, eta=0.0):
        """
        DDIM 单步采样
        """
        alpha_cumprod_t = self._extract(self.alphas_cumprod, t, x.shape)
        alpha_cumprod_t_prev = self._extract(self.alphas_cumprod_prev, t, x.shape)

        sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
        sqrt_alpha_cumprod_t_prev = torch.sqrt(alpha_cumprod_t_prev)
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alpha_cumprod_t)

        # 预测噪声（传入mask和angle）
        eps_theta = model(x, mask, angle=angle, t=t)

        x0_pred = (x - sqrt_one_minus_alpha_cumprod_t * eps_theta) / sqrt_alpha_cumprod_t
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        sigma = eta * torch.sqrt(
            (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) *
            (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
        )

        noise = torch.randn_like(x) if eta > 0 else 0
        dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigma ** 2) * eps_theta
        x_prev = sqrt_alpha_cumprod_t_prev * x0_pred + dir_xt + sigma * noise

        return x_prev

    @torch.no_grad()
    def sample_ddim(self, model, image_size, batch_size=16, channels=1,
                    ddim_steps=50, eta=0.0, mask=None, angle=None, progress=True):
        """
        DDIM 快速采样（支持条件）
        """
        shape = (batch_size, channels, image_size, image_size)
        img = torch.randn(shape, device=self.device)

        # 复制条件到batch维度
        if mask is not None and mask.shape[0] != batch_size:
            if mask.dim() == 4:
                mask = mask.repeat(batch_size, 1, 1, 1)
            elif mask.dim() == 5:
                mask = mask.repeat(batch_size, 1, 1, 1, 1)

        if angle is not None and angle.shape[0] != batch_size:
            angle = angle.repeat(batch_size, 1)

        intermediates = []
        ddim_timesteps = np.linspace(0, self.timesteps - 1, ddim_steps, dtype=int)[::-1]

        if progress:
            pbar = tqdm(range(len(ddim_timesteps)), desc="DDIM Sampling")

        for i, step in enumerate(ddim_timesteps):
            t = torch.full((batch_size,), step, device=self.device, dtype=torch.long)
            img = self.sample_timestep_ddim(model, img, t, mask=mask, angle=angle, eta=eta)
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
               mask=None, angle=None, progress=True):
        """
        统一的采样接口
        """
        if sampler_type == "ddpm":
            return self.sample_ddpm(model, image_size, batch_size, channels,
                                    mask=mask, angle=angle, progress=progress)
        elif sampler_type == "ddim":
            return self.sample_ddim(model, image_size, batch_size, channels,
                                    ddim_steps, eta, mask=mask, angle=angle, progress=progress)
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")

    # 添加lpips
    def predict_x0_from_noise(self, x_t, noise_pred, t):
        """
        从噪声图像和预测的噪声还原x0

        Args:
            x_t: 噪声图像 [B, C, H, W]
            noise_pred: 预测的噪声 [B, C, H, W]
            t: 时间步 [B]

        Returns:
            x0_pred: 预测的干净图像 [B, C, H, W]
        """
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

        x0_pred = (x_t - sqrt_one_minus_alphas_cumprod_t * noise_pred) / sqrt_alphas_cumprod_t
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)  # 限制范围

        return x0_pred
