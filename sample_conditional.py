"""
条件采样脚本：使用给定的血管掩码生成肺动脉造影图像
支持单张掩码生成和批量生成
"""

import os
import argparse
import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from config import Config
from utils.noise_schedule import get_noise_schedule
from utils.diffusion import GaussianDiffusion
from PARSE.mip import load_mask_and_mip
from models.cond_unet import ConditionalUNet


def load_model(checkpoint_path, config, device):
    """加载训练好的模型"""
    model = ConditionalUNet(
        in_channels=config.channels,
        out_channels=config.channels,
        base_channels=config.base_channels,
        cond_dim=config.cond_dim,
        time_emb_dim=config.time_emb_dim,
        block_type=config.cond_block_type
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"加载模型: {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"  训练轮次: {checkpoint['epoch']}")
    if 'loss' in checkpoint:
        print(f"  训练损失: {checkpoint['loss']:.6f}")

    return model


@torch.no_grad()
def sample_with_mask(model, diffusion, mask_path, config, device, num_samples=4, save_dir=None):
    """
    使用给定的3D掩码生成CT图像

    Args:
        model: 条件UNet
        diffusion: 扩散过程
        mask_path: 3D掩码文件路径 (.nii.gz)
        config: 配置
        device: 设备
        num_samples: 生成数量（使用相同条件生成多个变体）
        save_dir: 保存目录

    Returns:
        samples: 生成的图像 [num_samples, 1, H, W]
        mask_mip: MIP后的2D掩码
    """
    # 加载掩码并做MIP投影
    print(f"\n加载掩码: {mask_path}")
    mask_mip = load_mask_and_mip(
        mask_path,
        target_size=(config.image_size, config.image_size),
        device=device,
        projection_axis=config.mip_projection_axis
    )
    print(f"  MIP掩码形状: {mask_mip.shape}")
    print(f"  掩码范围: [{mask_mip.min():.3f}, {mask_mip.max():.3f}]")

    # 复制到batch维度
    mask_batch = mask_mip.repeat(num_samples, 1, 1, 1)

    # 生成样本
    print(f"\n生成 {num_samples} 个样本...")

    if config.sampler_type == "ddpm":
        print(f"使用 DDPM 采样 ({config.timesteps} 步)")
    else:
        print(f"使用 DDIM 采样 ({config.ddim_steps} 步, eta={config.ddim_eta})")

    # 条件采样循环
    shape = (num_samples, config.channels, config.image_size, config.image_size)
    img = torch.randn(shape, device=device)

    if config.sampler_type == "ddpm":
        # DDPM采样
        indices = list(range(config.timesteps))[::-1]

        from tqdm import tqdm
        for i in tqdm(indices, desc="DDPM采样"):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            predicted_noise = model(img, mask_batch, t)

            # 计算均值
            betas_t = diffusion._extract(diffusion.betas, t, img.shape)
            sqrt_one_minus_alphas_cumprod_t = diffusion._extract(
                diffusion.sqrt_one_minus_alphas_cumprod, t, img.shape
            )
            sqrt_recip_alphas_t = 1.0 / torch.sqrt(diffusion._extract(diffusion.alphas, t, img.shape))

            model_mean = sqrt_recip_alphas_t * (
                    img - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
            )

            if i == 0:
                img = model_mean
            else:
                noise = torch.randn_like(img)
                posterior_variance = betas_t
                img = model_mean + torch.sqrt(posterior_variance) * noise

    elif config.sampler_type == "ddim":
        # DDIM采样
        import numpy as np
        ddim_timesteps = np.linspace(0, config.timesteps - 1, config.ddim_steps, dtype=int)[::-1]

        from tqdm import tqdm
        for i, step in enumerate(tqdm(ddim_timesteps, desc="DDIM采样")):
            t = torch.full((num_samples,), step, device=device, dtype=torch.long)

            # 获取参数
            alpha_cumprod_t = diffusion._extract(diffusion.alphas_cumprod, t, img.shape)
            alpha_cumprod_t_prev = diffusion._extract(diffusion.alphas_cumprod_prev, t, img.shape)

            sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
            sqrt_alpha_cumprod_t_prev = torch.sqrt(alpha_cumprod_t_prev)
            sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alpha_cumprod_t)

            # 预测噪声
            eps_theta = model(img, mask_batch, t)

            # 预测 x0
            x0_pred = (img - sqrt_one_minus_alpha_cumprod_t * eps_theta) / sqrt_alpha_cumprod_t
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

            # 计算 sigma
            sigma = config.ddim_eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) *
                (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
            )

            # 生成噪声
            noise = torch.randn_like(img) if config.ddim_eta > 0 else 0

            # 计算方向
            dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigma ** 2) * eps_theta

            # 更新
            img = sqrt_alpha_cumprod_t_prev * x0_pred + dir_xt + sigma * noise

    # 反归一化
    samples = (img + 1) / 2
    samples = torch.clamp(samples, 0, 1)

    # 保存
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        # 保存单个图像
        for i in range(num_samples):
            save_path = os.path.join(save_dir, f"sample_{i}.png")
            save_image(samples[i], save_path)

        # 保存网格图
        import numpy as np
        grid_path = os.path.join(save_dir, "grid.png")
        save_image(samples, grid_path, nrow=int(np.ceil(np.sqrt(num_samples))))

        # 保存对比图（掩码 vs 生成结果）
        fig, axes = plt.subplots(2, min(num_samples, 8), figsize=(2 * min(num_samples, 8), 4))
        if min(num_samples, 8) == 1:
            axes = axes.reshape(-1, 1)

        for i in range(min(num_samples, 8)):
            axes[0, i].imshow(mask_mip[0].cpu().numpy(), cmap='hot')
            axes[0, i].set_title("Condition (MIP)", fontsize=8)
            axes[0, i].axis('off')

            axes[1, i].imshow(samples[i, 0].cpu().numpy(), cmap='gray')
            axes[1, i].set_title(f"Generated {i + 1}", fontsize=8)
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "comparison.png"), dpi=150)
        plt.close()

        print(f"\n结果保存至: {save_dir}")

    return samples, mask_mip


def main():
    parser = argparse.ArgumentParser(description="条件采样：使用血管掩码生成肺动脉造影图像")
    parser.add_argument("--mask_path", type=str, required=True, help="3D掩码文件路径 (.nii.gz)")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints_ct_cond/best_model.pt",
                        help="模型检查点路径")
    parser.add_argument("--num_samples", type=int, default=4, help="生成样本数量")
    parser.add_argument("--output_dir", type=str, default="./generated_cond", help="输出目录")
    parser.add_argument("--device", type=str, default=None, help="设备 (cuda/cpu)")
    args = parser.parse_args()

    # 加载配置
    config = Config()

    # 设置设备
    if args.device:
        device = args.device
    else:
        device = config.device

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("条件采样 - 肺动脉造影生成")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Sampler: {config.sampler_type}")
    print(f"Image size: {config.image_size}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 60)

    # 加载模型
    model = load_model(args.checkpoint, config, device)

    # 初始化扩散过程（用于参数）
    betas = get_noise_schedule(config)
    diffusion = GaussianDiffusion(betas, device)

    # 生成样本
    samples, mask_mip = sample_with_mask(
        model=model,
        diffusion=diffusion,
        mask_path=args.mask_path,
        config=config,
        device=device,
        num_samples=args.num_samples,
        save_dir=args.output_dir
    )

    print("\n" + "=" * 60)
    print("采样完成！")
    print(f"生成图像: {args.output_dir}/sample_*.png")
    print(f"网格图: {args.output_dir}/grid.png")
    print(f"对比图: {args.output_dir}/comparison.png")
    print("=" * 60)


if __name__ == "__main__":
    main()