"""
条件采样脚本：使用给定的血管掩码生成肺动脉造影图像
支持2D MIP掩码和3D原始掩码
直接运行或修改下方配置参数即可
"""

import os
import numpy as np
import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import platform

from config import Config
from utils.noise_schedule import get_noise_schedule
from utils.diffusion import GaussianDiffusion
from utils.analyse_angle import get_angle_info
from models.cond_unet import ConditionalUNet

# macOS 特定设置
if platform.system() == "Darwin":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==================== 配置参数（直接在这里修改） ====================
# 基础配置
MASK_PATH = "./data/mask/patient001.nii.gz"  # 掩码文件路径（.nii.gz 或 .npy）
CHECKPOINT_PATH = "./checkpoints_ct_3d_angle/best_model.pt"  # 模型检查点路径
OUTPUT_DIR = "./generated_samples"  # 输出目录

# 采样配置
NUM_SAMPLES = 4  # 生成样本数量
SPECIFIC_ANGLE = None  # 指定生成角度（度），None表示从文件名解析，例如: 45
USE_SAMPLER = None  # 采样器类型，None使用配置文件设置，可选: "ddpm", "ddim"
DEVICE = None  # 设备，None自动选择，可选: "cuda", "cpu"

# 可视化配置
SHOW_COMPARISON = True  # 是否显示对比图
SHOW_PROCESS = True  # 是否显示生成过程
SAVE_INDIVIDUAL = True  # 是否保存单个图像
# ==================================================================


def load_model(checkpoint_path, config, device):
    """加载训练好的模型"""
    model = ConditionalUNet(
        in_channels=config.channels,
        out_channels=config.channels,
        base_channels=config.base_channels,
        cond_dim=config.cond_dim,
        time_emb_dim=config.time_emb_dim,
        block_type=config.cond_block_type,
        mask_type=config.mask_type,
        use_angle=config.use_angle_condition,
        angle_dim=config.angle_dim
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ 加载模型: {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"  训练轮次: {checkpoint['epoch']}")
    if 'loss' in checkpoint:
        print(f"  训练损失: {checkpoint['loss']:.6f}")

    return model


def load_3d_mask(mask_path, target_size=(64, 64, 64), device='cpu'):
    """加载3D掩码（.nii.gz）"""
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("需要安装 nibabel: pip install nibabel")

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
        try:
            from scipy.ndimage import zoom
        except ImportError:
            raise ImportError("需要安装 scipy: pip install scipy")

        zoom_factors = (target_size[0] / mip_array.shape[0],
                        target_size[1] / mip_array.shape[1])
        mip_array = zoom(mip_array, zoom_factors, order=1)

    mask_tensor = torch.from_numpy(mip_array).float().unsqueeze(0)  # [1, H, W]
    return mask_tensor.to(device)


def parse_angle_from_mask_path(mask_path, config):
    """从掩码文件路径解析角度信息"""
    filename = os.path.basename(mask_path)

    if config.mask_type == "3d":
        print(f"⚠ 3D掩码文件名 '{filename}' 不包含角度信息，使用默认角度 0°")
        angle_info = get_angle_info("patient_0_mask.png", angle_rep=config.angle_rep)
    else:
        base_name = filename.replace('_mip.npy', '')
        virtual_filename = f"{base_name}_0_mask.png"
        print(f"⚠ 2D MIP文件名 '{filename}' 不包含角度信息，使用默认角度 0°")
        angle_info = get_angle_info(virtual_filename, angle_rep=config.angle_rep)

    angle_tensor = torch.from_numpy(angle_info['angle_vector']).float()

    print(f"  解析角度: {angle_info['angle_deg']}°")
    print(f"  角度表示: {config.angle_rep}")

    return angle_tensor, angle_info


def visualize_mask(mask_tensor, mask_type):
    """可视化掩码（返回适合imshow的numpy数组）"""
    mask_np = mask_tensor[0].cpu().numpy()

    if mask_type == "2d":
        return mask_np
    else:
        # 3D掩码：取中间切片
        mid_slice = mask_np.shape[0] // 2
        return mask_np[mid_slice]


@torch.no_grad()
def sample_with_mask(model, diffusion, mask_path, config, device, num_samples=4,
                     save_dir=None, specific_angle=None):
    """使用给定的掩码生成图像"""

    # 根据类型加载掩码
    print(f"\n📂 加载掩码: {mask_path}")
    if config.mask_type == "2d":
        mask = load_2d_mask(
            mask_path,
            target_size=(config.image_size, config.image_size),
            device=device
        )
        print(f"  2D MIP掩码形状: {mask.shape}")
        mask_batch = mask.repeat(num_samples, 1, 1, 1)
    else:
        mask = load_3d_mask(
            mask_path,
            target_size=config.mask_3d_size,
            device=device
        )
        print(f"  3D掩码形状: {mask.shape}")
        mask_batch = mask.repeat(num_samples, 1, 1, 1, 1)

    print(f"  掩码范围: [{mask.min():.3f}, {mask.max():.3f}]")

    # 解析角度信息
    print(f"\n📐 解析角度信息:")
    if specific_angle is not None:
        import math
        angle_rad = math.radians(specific_angle)
        from utils.analyse_angle import euler_to_quaternion

        if config.angle_rep == "quaternion":
            angle_vector = euler_to_quaternion(angle_rad)
        else:
            angle_vector = np.array([0.0, 0.0, angle_rad], dtype=np.float32)

        angle = torch.from_numpy(angle_vector).float().to(device)
        print(f"  使用指定角度: {specific_angle}°")
    else:
        angle, angle_info = parse_angle_from_mask_path(mask_path, config)
        angle = angle.to(device)

    angle_batch = angle.repeat(num_samples, 1)

    # 生成样本
    print(f"\n🎨 生成 {num_samples} 个样本...")
    print(f"  采样器: {config.sampler_type}")
    if config.sampler_type == "ddim":
        print(f"  DDIM步数: {config.ddim_steps}, eta: {config.ddim_eta}")

    samples, intermediates = diffusion.sample(
        model,
        config.image_size,
        batch_size=num_samples,
        channels=config.channels,
        sampler_type=config.sampler_type,
        ddim_steps=config.ddim_steps,
        eta=config.ddim_eta,
        mask=mask_batch,
        angle=angle_batch,
        progress=True
    )

    # 反归一化
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)

    # 保存结果
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        # 保存单个图像
        if SAVE_INDIVIDUAL:
            for i in range(num_samples):
                save_path = os.path.join(save_dir, f"sample_{i}.png")
                save_image(samples[i], save_path)
            print(f"  ✓ 保存 {num_samples} 个单独图像")

        # 保存网格图
        grid_path = os.path.join(save_dir, "grid.png")
        save_image(samples, grid_path, nrow=int(np.ceil(np.sqrt(num_samples))))
        print(f"  ✓ 保存网格图: grid.png")

        # 保存对比图
        if SHOW_COMPARISON:
            display_num = min(8, num_samples)
            fig, axes = plt.subplots(2, display_num, figsize=(2 * display_num, 4))
            if display_num == 1:
                axes = axes.reshape(-1, 1)

            for i in range(display_num):
                # 第一行：条件掩码
                mask_disp = visualize_mask(mask, config.mask_type)
                axes[0, i].imshow(mask_disp, cmap='hot')

                angle_text = f"{specific_angle}°" if specific_angle is not None else f"{angle_info['angle_deg']}°"
                axes[0, i].set_title(f"Mask + {angle_text}", fontsize=8)
                axes[0, i].axis('off')

                # 第二行：生成结果
                sample_disp = samples[i, 0].cpu().numpy()
                axes[1, i].imshow(sample_disp, cmap='gray')
                axes[1, i].set_title(f"Sample {i+1}", fontsize=8)
                axes[1, i].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "comparison.png"), dpi=150)
            plt.close()
            print(f"  ✓ 保存对比图: comparison.png")

        # 保存生成过程
        if SHOW_PROCESS and len(intermediates) > 0:
            num_steps_to_show = min(5, len(intermediates))
            fig, axes = plt.subplots(1, num_steps_to_show,
                                      figsize=(3 * num_steps_to_show, 3))
            if num_steps_to_show == 1:
                axes = [axes]

            step_indices = np.linspace(0, len(intermediates)-1, num_steps_to_show, dtype=int)
            for idx, step_idx in enumerate(step_indices):
                img = intermediates[step_idx]
                img_display = (img[0, 0] + 1) / 2
                axes[idx].imshow(img_display.cpu().numpy(), cmap='gray')
                axes[idx].axis('off')

                if config.sampler_type == "ddpm":
                    step_num = (len(intermediates) - 1 - step_idx) * (config.timesteps // len(intermediates))
                else:
                    step_num = (len(intermediates) - 1 - step_idx) * (config.ddim_steps // len(intermediates))
                axes[idx].set_title(f'Step {step_num}', fontsize=10)

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "process.png"), dpi=150)
            plt.close()
            print(f"  ✓ 保存生成过程: process.png")

        # 保存元数据
        import json
        metadata = {
            'mask_path': mask_path,
            'mask_type': config.mask_type,
            'num_samples': num_samples,
            'sampler_type': config.sampler_type,
            'ddim_steps': config.ddim_steps if config.sampler_type == "ddim" else None,
            'image_size': config.image_size,
            'angle_value': float(specific_angle) if specific_angle is not None else float(angle_info['angle_deg'])
        }

        with open(os.path.join(save_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"  ✓ 保存元数据: metadata.json")

    return samples, mask


def main():
    """主函数 - 直接运行即可"""

    print("=" * 60)
    print("🎯 条件扩散模型 - 肺动脉造影生成")
    print("=" * 60)

    # 加载配置
    config = Config()

    # 应用用户配置
    if USE_SAMPLER:
        config.sampler_type = USE_SAMPLER

    device = DEVICE if DEVICE else config.device

    # 打印配置信息
    print(f"\n⚙️  运行配置:")
    print(f"  设备: {device}")
    print(f"  掩码文件: {MASK_PATH}")
    print(f"  模型检查点: {CHECKPOINT_PATH}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  样本数量: {NUM_SAMPLES}")
    print(f"  掩码类型: {config.mask_type}")
    print(f"  采样器: {config.sampler_type}")
    if SPECIFIC_ANGLE:
        print(f"  指定角度: {SPECIFIC_ANGLE}°")
    else:
        print(f"  角度: 从文件名自动解析")

    # 检查文件是否存在
    if not os.path.exists(MASK_PATH):
        print(f"\n❌ 错误: 掩码文件不存在!")
        print(f"   路径: {MASK_PATH}")
        print(f"   请修改 MASK_PATH 变量为正确的文件路径")
        return

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\n❌ 错误: 模型检查点不存在!")
        print(f"   路径: {CHECKPOINT_PATH}")
        print(f"   请修改 CHECKPOINT_PATH 变量为正确的检查点路径")
        return

    try:
        # 加载模型
        print(f"\n🔧 加载模型...")
        model = load_model(CHECKPOINT_PATH, config, device)

        # 初始化扩散过程
        betas = get_noise_schedule(config)
        diffusion = GaussianDiffusion(betas, device)

        # 生成样本
        samples, mask = sample_with_mask(
            model=model,
            diffusion=diffusion,
            mask_path=MASK_PATH,
            config=config,
            device=device,
            num_samples=NUM_SAMPLES,
            save_dir=OUTPUT_DIR,
            specific_angle=SPECIFIC_ANGLE
        )

        print("\n" + "=" * 60)
        print("✅ 生成完成!")
        print("=" * 60)
        print(f"\n📁 结果保存在: {OUTPUT_DIR}/")
        if SAVE_INDIVIDUAL:
            print(f"  - 单独图像: sample_*.png")
        print(f"  - 网格图: grid.png")
        if SHOW_COMPARISON:
            print(f"  - 对比图: comparison.png")
        if SHOW_PROCESS:
            print(f"  - 生成过程: process.png")
        print(f"  - 元数据: metadata.json")

        # 尝试自动打开输出文件夹
        try:
            import subprocess
            if platform.system() == "Windows":
                os.startfile(OUTPUT_DIR)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", OUTPUT_DIR])
            else:  # Linux
                subprocess.run(["xdg-open", OUTPUT_DIR])
            print(f"\n📂 已自动打开输出文件夹")
        except:
            pass

    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 提示:")
        print("  1. 检查掩码文件路径是否正确")
        print("  2. 检查模型检查点是否完整")
        print("  3. 确保已安装所有依赖: nibabel, scipy")


if __name__ == "__main__":
    # 直接运行即可！
    main()

    # ==================== 使用示例 ====================
    # 示例1: 生成单个掩码的多个视角
    # MASK_PATH = "./data/mask/patient001.nii.gz"
    # NUM_SAMPLES = 4
    # SPECIFIC_ANGLE = 45  # 生成45度视角

    # 示例2: 批量生成不同角度（需要手动修改循环）
    # for angle in [0, 22.5, 45, 67.5, 90]:
    #     SPECIFIC_ANGLE = angle
    #     OUTPUT_DIR = f"./generated_angle_{angle}"
    #     main()

    # 示例3: 处理多个患者
    # patients = ["patient001", "patient002", "patient003"]
    # for patient in patients:
    #     MASK_PATH = f"./data/mask/{patient}.nii.gz"
    #     OUTPUT_DIR = f"./results/{patient}"
    #     main()