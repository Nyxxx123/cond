"""
扩散过程核心实现
支持 DDPM 和 DDIM 两种采样方式
支持条件生成
支持 ε-prediction 和 v-prediction 两种预测类型
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
    支持 prediction_type: "epsilon" 或 "v"
    """
    def __init__(self, betas, device, prediction_type="epsilon"):
        """
        betas: 噪声调度序列
        device: 计算设备
        prediction_type: "epsilon" 或 "v"
        """
        self.betas = betas.to(device)
        self.device = device
        self.timesteps = len(betas)
        self.prediction_type = prediction_type

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

        # v-prediction 需要的角度参数: φ_t = arccos(√ᾱ_t)
        # 则 cosφ_t = √ᾱ_t, sinφ_t = √(1-ᾱ_t)
        self.cos_phi = self.sqrt_alphas_cumprod
        self.sin_phi = self.sqrt_one_minus_alphas_cumprod

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

    def compute_target(self, x_start, t, noise):
        """
        根据 prediction_type 计算训练目标
        """
        if self.prediction_type == "epsilon":
            return noise
        elif self.prediction_type == "v":
            sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
            sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            v_target = sqrt_alphas_cumprod_t * noise - sqrt_one_minus_alphas_cumprod_t * x_start
            return v_target
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

    def predict_x0_and_eps_from_v(self, x_t, v_pred, t):
        """
        从 v-prediction 恢复 x0_pred 和 eps_pred
        """
        cos_phi_t = self._extract(self.cos_phi, t, x_t.shape)
        sin_phi_t = self._extract(self.sin_phi, t, x_t.shape)

        x0_pred = cos_phi_t * x_t - sin_phi_t * v_pred
        eps_pred = sin_phi_t * x_t + cos_phi_t * v_pred

        return x0_pred, eps_pred

    def predict_noise_from_x0(self, x_t, x0_pred, t):
        """
        从预测的 x0 恢复预测的噪声（用于 ε-prediction 模式）
        """
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

        eps_pred = (x_t - sqrt_alphas_cumprod_t * x0_pred) / sqrt_one_minus_alphas_cumprod_t
        return eps_pred

    def p_losses(self, model, x_start, t, mask=None, angle=None, non_angio=None, noise=None):
        """
        计算训练损失（支持 ε-prediction 和 v-prediction）
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start, t, noise)

        # 模型预测（输出与预测目标维度相同）
        # ========== 新增 non_angio 参数 ==========
        model_output = model(x_noisy, mask, angle=angle, non_angio=non_angio, t=t)
        # ======================================

        # 计算目标值
        target = self.compute_target(x_start, t, noise)

        # 计算损失
        loss = F.mse_loss(model_output, target)

        return loss

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index, mask=None, angle=None, non_angio=None):
        """
        单步逆向采样（支持 ε-prediction 和 v-prediction）
        """
        t = t.long().to(self.device)

        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = 1.0 / torch.sqrt(self._extract(self.alphas, t, x.shape))

        # 模型预测
        # ========== 新增 non_angio 参数 ==========
        model_output = model(x, mask, angle=angle, non_angio=non_angio, t=t)
        # ======================================

        if self.prediction_type == "epsilon":
            predicted_noise = model_output
            model_mean = sqrt_recip_alphas_t * (
                x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
            )
        else:  # v-prediction
            # 从 v 恢复 x0_pred 和 eps_pred
            cos_phi_t = self._extract(self.cos_phi, t, x.shape)
            sin_phi_t = self._extract(self.sin_phi, t, x.shape)

            x0_pred = cos_phi_t * x - sin_phi_t * model_output
            eps_pred = sin_phi_t * x + cos_phi_t * model_output

            # 使用与 ε-prediction 相同的更新公式
            model_mean = sqrt_recip_alphas_t * (
                x - betas_t * eps_pred / sqrt_one_minus_alphas_cumprod_t
            )

        if t_index == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            posterior_variance = betas_t
            return model_mean + torch.sqrt(posterior_variance) * noise

    @torch.no_grad()
    def sample_ddpm(self, model, image_size, batch_size=16, channels=1,
                    mask=None, angle=None, non_angio=None, progress=True):
        """
        完整DDPM采样循环
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

        # ========== 新增 non_angio 的复制 ==========
        if non_angio is not None and non_angio.shape[0] != batch_size:
            non_angio = non_angio.repeat(batch_size, 1, 1, 1)
        # ========================================

        intermediates = []
        indices = list(range(self.timesteps))[::-1]

        if progress:
            indices = tqdm(indices, desc="DDPM Sampling")

        for i in indices:
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            # ========== 新增 non_angio 参数 ==========
            img = self.p_sample(model, img, t, i, mask=mask, angle=angle, non_angio=non_angio)
            # ======================================

            if i % 100 == 0 or i == self.timesteps - 1 or i == 0:
                intermediates.append(img.cpu())

        return img, intermediates

    @torch.no_grad()
    def sample_timestep_ddim(self, model, x, t, t_prev, mask=None, angle=None, non_angio=None, eta=0.0):
        """
        DDIM 单步采样（支持 ε-prediction 和 v-prediction）

        t: 当前DDIM子序列步（用于查 ᾱ_t）
        t_prev: DDIM子序列的下一步（用于查 ᾱ_{t_prev}），必须传入，不能靠 alphas_cumprod_prev
        """
        alpha_cumprod_t = self._extract(self.alphas_cumprod, t, x.shape)
        alpha_cumprod_t_prev = self._extract(self.alphas_cumprod, t_prev, x.shape)

        sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
        sqrt_alpha_cumprod_t_prev = torch.sqrt(alpha_cumprod_t_prev)
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alpha_cumprod_t)

        # 模型预测（时间嵌入使用当前步 t）
        # ========== 新增 non_angio 参数 ==========
        model_output = model(x, mask, angle=angle, non_angio=non_angio, t=t)
        # ======================================

        if self.prediction_type == "epsilon":
            eps_theta = model_output
            x0_pred = (x - sqrt_one_minus_alpha_cumprod_t * eps_theta) / sqrt_alpha_cumprod_t
        else:  # v-prediction
            cos_phi_t = self._extract(self.cos_phi, t, x.shape)
            sin_phi_t = self._extract(self.sin_phi, t, x.shape)
            x0_pred = cos_phi_t * x - sin_phi_t * model_output
            eps_theta = sin_phi_t * x + cos_phi_t * model_output

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
                    ddim_steps=50, eta=0.0, mask=None, angle=None, non_angio=None, progress=True):
        """
        DDIM 快速采样
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

        # ========== 新增 non_angio 的复制 ==========
        if non_angio is not None and non_angio.shape[0] != batch_size:
            non_angio = non_angio.repeat(batch_size, 1, 1, 1)
        # ========================================

        intermediates = []
        # 生成 DDIM 时间步子序列，从大到小排列
        ddim_timesteps = np.linspace(0, self.timesteps - 1, ddim_steps, dtype=int)[::-1]

        if progress:
            pbar = tqdm(range(len(ddim_timesteps) - 1), desc="DDIM Sampling")

        # 注意：只迭代 len-1 次。第 i 次从 τ_i 跳到 τ_{i+1}
        for i in range(len(ddim_timesteps) - 1):
            step_curr = ddim_timesteps[i]      # 当前噪声水平 τ_i
            step_next = ddim_timesteps[i + 1]  # 目标噪声水平 τ_{i+1}

            t_curr = torch.full((batch_size,), step_curr, device=self.device, dtype=torch.long)
            t_next = torch.full((batch_size,), step_next, device=self.device, dtype=torch.long)

            # ========== 新增 non_angio 参数 ==========
            img = self.sample_timestep_ddim(model, img, t_curr, t_next,
                                            mask=mask, angle=angle, non_angio=non_angio, eta=eta)
            # ======================================
            img = torch.clamp(img, -1.0, 1.0)

            if i % max(1, (len(ddim_timesteps) - 1) // 10) == 0 or i == len(ddim_timesteps) - 2:
                intermediates.append(img.cpu())

            if progress:
                pbar.update(1)

        if progress:
            pbar.close()

        return img, intermediates

    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=1,
               sampler_type="ddpm", ddim_steps=50, eta=0.0,
               mask=None, angle=None, non_angio=None, progress=True):
        """
        统一的采样接口
        """
        if sampler_type == "ddpm":
            return self.sample_ddpm(model, image_size, batch_size, channels,
                                    mask=mask, angle=angle, non_angio=non_angio, progress=progress)
        elif sampler_type == "ddim":
            return self.sample_ddim(model, image_size, batch_size, channels,
                                    ddim_steps, eta, mask=mask, angle=angle, non_angio=non_angio, progress=progress)
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")

    def predict_x0_from_noise(self, x_t, noise_pred, t):
        """
        从噪声图像和预测的噪声还原x0（用于 ε-prediction 模式）
        """
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

        x0_pred = (x_t - sqrt_one_minus_alphas_cumprod_t * noise_pred) / sqrt_alphas_cumprod_t
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        return x0_pred

    def predict_x0_from_v(self, x_t, v_pred, t):
        """
        从速度预测还原x0（用于 v-prediction 模式）
        注意：此方法用于训练时计算 LPIPS，不需要 clamp
        """
        cos_phi_t = self._extract(self.cos_phi, t, x_t.shape)
        sin_phi_t = self._extract(self.sin_phi, t, x_t.shape)
        x0_pred = cos_phi_t * x_t - sin_phi_t * v_pred
        return x0_pred

    @torch.no_grad()
    def refine_x0_ddim(self, model, x_t, t, mask=None, angle=None, non_angio=None,
                       refine_steps=5, eta=0.0):
        """
        用少量 DDIM 子步从 x_t 精修出更接近最终推理质量的 x_0。

        训练时单步 predict_x0 在 t 较大时非常粗糙，
        而推理时使用完整 DDPM/DDIM 多步采样，二者质量差距明显。
        此方法在 [t-1, 0] 区间内均匀取 refine_steps 个子步，
        执行确定性 DDIM 子序列采样，缩小训练-推理 gap。
        """
        if refine_steps <= 1:
            return x_t

        t_int = t[0].item()
        if t_int < 2:
            # t=0 或 t=1 时 x_t 已接近 x_0，无需精修；且 t_int-1 会产生无效索引
            return x_t

        sub_steps = np.linspace(t_int - 1, 0, refine_steps, dtype=int)

        x = x_t
        for i in range(len(sub_steps) - 1):
            t_curr = torch.full_like(t, sub_steps[i])
            t_next = torch.full_like(t, sub_steps[i + 1])
            x = self.sample_timestep_ddim(
                model, x, t_curr, t_next,
                mask=mask, angle=angle, non_angio=non_angio, eta=eta
            )
            x = torch.clamp(x, -1.0, 1.0)

        return x