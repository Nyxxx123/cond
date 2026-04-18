"""
条件采样脚本：使用给定的血管掩码生成肺动脉造影图像
支持2D MIP掩码和3D原始掩码
"""

import os
import argparse
import numpy as np
import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from config import Config
from utils.noise_schedule import get_noise_schedule
from utils.diffusion import GaussianDiffusion
from models.cond_unet import ConditionalUNet


def load_model(checkpoint_path, config, device):
    """加载训练好的模型"""
    model = ConditionalUNet(
        in_channels=config.channels,
        out_channels=config.channels,
        base_channels=config.base_channels,
        cond_dim=config.cond_dim,
        time_emb_dim=config.time_emb_dim,
        block_type=config.cond_block_type,
        mask_type=config.mask_type  # 新增这一行
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


def load_3d_mask(mask_path, target_size=(64, 64, 64), device='cpu'):
    """加载3D掩码（.nii.gz）"""
    import nibabel as nib
    nii = nib.load(mask_path)
    mask_3d = nii.get_fdata()
    mask_3d = mask_3d[np.newaxis, ...]  # [1, D, H, W]
    mask_tensor = torch.from_numpy(mask_3d).float()

    # 下采样到目标尺寸
    if mask_tensor.shape[1:] != target_size:
        mask_tensor = mask_tensor.unsqueeze(0)
        mask_tensor = torch.nn.functional.interpolate(
            mask_tensor,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
        mask_tensor = mask_tensor.squeeze(0)

    # 归一化到[0,1]
    if mask_tensor.max() > 0:
        mask_tensor = mask_tensor / mask_tensor.max()

    return mask_tensor.to(device)


def load_2d_mask(mask_path, target_size=(256, 256), device='cpu'):
    """加载2D MIP掩码（.npy）"""
    mip_array = np.load(mask_path)

    # 调整尺寸
    if mip_array.shape != target_size:
        from scipy.ndimage import zoom
        zoom_factors = (target_size[0] / mip_array.shape[0],
                        target_size[1] / mip_array.shape[1])
        mip_array = zoom(mip_array, zoom_factors, order=1)

    mask_tensor = torch.from_numpy(mip_array).float().unsqueeze(0)  # [1, H, W]
    return mask_tensor.to(device)


def visualize_mask(mask_tensor, mask_type):
    """
    可视化掩码（返回适合imshow的numpy数组）
    """
    mask_np = mask_tensor[0].cpu().numpy()

    if mask_type == "2d":
        return mask_np
    else:
        # 3D掩码：取中间切片
        mid_slice = mask_np.shape[0] // 2
        return mask_np[mid_slice]


@torch.no_grad()
def sample_with_mask(model, diffusion, mask_path, config, device, num_samples=4, save_dir=None):
    """
    使用给定的掩码生成图像
    """
    # 根据类型加载掩码
    print(f"\n加载掩码: {mask_path}")
    if config.mask_type == "2d":
        mask = load_2d_mask(
            mask_path,
            target_size=(config.image_size, config.image_size),
            device=device
        )
        print(f"  2D MIP掩码形状: {mask.shape}")
        # 复制到batch维度
        mask_batch = mask.repeat(num_samples, 1, 1, 1)
    else:
        mask = load_3d_mask(
            mask_path,
            target_size=config.mask_3d_size,
            device=device
        )
        print(f"  3D掩码形状: {mask.shape}")
        # 复制到batch维度
        mask_batch = mask.repeat(num_samples, 1, 1, 1, 1)

    print(f"  掩码范围: [{mask.min():.3f}, {mask.max():.3f}]")

    # 生成样本
    print(f"\n生成 {num_samples} 个样本...")
    print(f"采样器: {config.sampler_type}")

    samples, intermediates = diffusion.sample(
        model,
        config.image_size,
        batch_size=num_samples,
        channels=config.channels,
        sampler_type=config.sampler_type,
        ddim_steps=config.ddim_steps,
        eta=config.ddim_eta,
        cond=mask_batch,
        progress=True
    )

    # 反归一化
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)

    # 保存结果
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        # 保存单个图像
        for i in range(num_samples):
            save_path = os.path.join(save_dir, f"sample_{i}.png")
            save_image(samples[i], save_path)

        # 保存网格图
        grid_path = os.path.join(save_dir, "grid.png")
        save_image(samples, grid_path, nrow=int(np.ceil(np.sqrt(num_samples))))

        # 保存对比图（掩码 vs 生成结果）
        display_num = min(8, num_samples)
        fig, axes = plt.subplots(2, display_num, figsize=(2 * display_num, 4))
        if display_num == 1:
            axes = axes.reshape(-1, 1)

        for i in range(display_num):
            # 第一行：条件掩码
            mask_disp = visualize_mask(mask, config.mask_type)
            axes[0, i].imshow(mask_disp, cmap='hot')
            axes[0, i].set_title(f"Condition ({config.mask_type})", fontsize=8)
            axes[0, i].axis('off')

            # 第二行：生成结果
            sample_disp = samples[i, 0].cpu().numpy()
            axes[1, i].imshow(sample_disp, cmap='gray')
            axes[1, i].set_title(f"Generated {i+1}", fontsize=8)
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "comparison.png"), dpi=150)
        plt.close()

        # 保存生成过程（可选）
        if len(intermediates) > 0:
            fig, axes = plt.subplots(1, min(10, len(intermediates)),
                                      figsize=(2 * min(10, len(intermediates)), 2))
            if min(10, len(intermediates)) == 1:
                axes = [axes]
            for idx, img in enumerate(intermediates[:10]):
                img_display = (img[0, 0] + 1) / 2
                axes[idx].imshow(img_display.numpy(), cmap='gray')
                axes[idx].axis('off')
                axes[idx].set_title(f'Step {idx * 100}')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "sampling_process.png"), dpi=150)
            plt.close()

        print(f"\n结果保存至: {save_dir}")

    return samples, mask


def main():
    parser = argparse.ArgumentParser(description="条件采样：使用血管掩码生成肺动脉造影图像")
    parser.add_argument("--mask_path", type=str, required=True,
                        help="掩码文件路径 (.nii.gz 用于3D, .npy 用于2D MIP)")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints_ct_cond/best_model.pt",
                        help="模型检查点路径")
    parser.add_argument("--num_samples", type=int, default=4,
                        help="生成样本数量")
    parser.add_argument("--output_dir", type=str, default="./generated_cond",
                        help="输出目录")
    parser.add_argument("--device", type=str, default=None,
                        help="设备 (cuda/cpu)")
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
    print(f"Mask type: {config.mask_type}")
    print(f"Sampler: {config.sampler_type}")
    print(f"Image size: {config.image_size}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 60)

    # 加载模型
    model = load_model(args.checkpoint, config, device)

    # 初始化扩散过程
    betas = get_noise_schedule(config)
    diffusion = GaussianDiffusion(betas, device)

    # 生成样本
    samples, mask = sample_with_mask(
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