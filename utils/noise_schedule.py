"""
噪声调度函数
"""

import torch
import math

def get_linear_noise_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """
    线性噪声调度
    """
    scale = 1000 / timesteps
    beta_start_scaled = scale * beta_start
    beta_end_scaled = scale * beta_end
    betas = torch.linspace(beta_start_scaled, beta_end_scaled, timesteps)
    return betas

def get_cosine_noise_schedule(timesteps, s=0.008):
    """
    余弦噪声调度
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas
    return torch.clip(betas, 0.0001, 0.9999)

def get_noise_schedule(config):
    """
    根据配置获取噪声调度
    """
    if config.schedule_type == "linear":
        return get_linear_noise_schedule(
            config.timesteps,
            config.beta_start,
            config.beta_end
        )
    elif config.schedule_type == "cosine":
        return get_cosine_noise_schedule(config.timesteps)
    else:
        raise ValueError(f"Unknown schedule type: {config.schedule_type}")