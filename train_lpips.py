"""
条件扩散模型训练脚本 - 肺动脉造影版本
使用血管掩码作为条件（支持2D MIP或3D原始掩码）
添加LPIPS感知损失提升生成质量
"""

import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
import lpips

from config import Config
from utils.noise_schedule import get_noise_schedule
from utils.diffusion import GaussianDiffusion
from utils.dataset import create_dataloader
from models.cond_unet import ConditionalUNet

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def prepare_data(shuffle=True):
    """准备条件数据集"""
    dataloader = create_dataloader(shuffle=shuffle)
    return dataloader


def train_one_epoch(model, diffusion, dataloader, optimizer, config, epoch, lpips_loss_fn):
    """训练一个epoch（包含LPIPS损失）"""
    model.train()
    total_loss = 0
    total_mse_loss = 0
    total_lpips_loss = 0
    num_batches = len(dataloader)
    batch_losses = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")
    for batch_idx, batch in enumerate(pbar):
        # 获取数据
        targets = batch['target'].to(config.device)
        masks = batch['mask'].to(config.device)
        angles = batch['angle'].to(config.device)
        batch_size = targets.shape[0]

        # 随机采样时间步
        t = torch.randint(0, config.timesteps, (batch_size,), device=config.device).long()

        # 前向扩散：添加噪声
        noise = torch.randn_like(targets)
        x_noisy = diffusion.q_sample(targets, t, noise)

        # 模型预测噪声
        predicted_noise = model(x_noisy, masks, angles, t)

        # ========== 损失计算 ==========
        # 1. MSE损失（主要损失）
        mse_loss = F.mse_loss(predicted_noise, noise)

        # 2. LPIPS感知损失（可选）
        if config.use_lpips:
            # 从预测噪声还原x0_pred
            x0_pred = diffusion.predict_x0_from_noise(x_noisy, predicted_noise, t)
            # 计算感知损失
            lpips_loss_val = lpips_loss_fn(x0_pred, targets).mean()
            # 组合损失
            loss = mse_loss + config.lpips_loss_weight * lpips_loss_val
            total_lpips_loss += lpips_loss_val.item()
        else:
            loss = mse_loss
            lpips_loss_val = torch.tensor(0.0)

        total_mse_loss += mse_loss.item()
        # ============================

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        batch_losses.append(loss.item())

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mse': f'{mse_loss.item():.4f}',
            'lpips': f'{lpips_loss_val.item():.4f}',
            'avg_loss': f'{avg_loss:.4f}'
        })

    avg_total_loss = total_loss / num_batches
    avg_mse_loss = total_mse_loss / num_batches
    avg_lpips_loss = total_lpips_loss / num_batches if config.use_lpips else 0

    return avg_total_loss, avg_mse_loss, avg_lpips_loss, batch_losses


def sample_and_save(model, diffusion, dataloader, config, epoch):
    """生成样本并保存 - 对比目标图像和生成图像"""
    model.eval()

    # 从dataloader中获取一个batch的数据
    for batch in dataloader:
        test_masks = batch['mask'][:config.sample_batch_size].to(config.device)
        test_angles = batch['angle'][:config.sample_batch_size].to(config.device)
        test_targets = batch['target'][:config.sample_batch_size]
        break

    actual_batch_size = test_masks.shape[0]
    num_samples = actual_batch_size

    with torch.no_grad():
        samples, intermediates = diffusion.sample(
            model,
            config.image_size,
            batch_size=num_samples,
            channels=config.channels,
            sampler_type=config.sampler_type,
            ddim_steps=config.ddim_steps,
            eta=config.ddim_eta,
            mask=test_masks,
            angle=test_angles,
            progress=True
        )

        # 反归一化到[0,1]
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)

        # 目标图像也需要反归一化（从[-1,1]到[0,1]）
        targets_disp = (test_targets + 1) / 2
        targets_disp = torch.clamp(targets_disp, 0, 1)

        save_path = os.path.join(config.sample_dir, f"epoch_{epoch + 1}.png")
        save_image(samples, save_path, nrow=4)

        # 保存对比图：目标图像 vs 生成图像
        display_num = min(8, num_samples)
        fig, axes = plt.subplots(2, display_num, figsize=(2 * display_num, 4))
        if display_num == 1:
            axes = axes.reshape(-1, 1)

        for i in range(display_num):
            # 第一行：目标图像（Ground Truth）
            target_disp = targets_disp[i, 0].cpu().numpy()
            axes[0, i].imshow(target_disp, cmap='gray')
            axes[0, i].set_title(f"Ground Truth", fontsize=8)
            axes[0, i].axis('off')

            # 第二行：生成图像
            sample_disp = samples[i, 0].cpu().numpy()
            axes[1, i].imshow(sample_disp, cmap='gray')
            axes[1, i].set_title(f"Generated", fontsize=8)
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(config.sample_dir, f"epoch_{epoch + 1}_comparison.png"), dpi=150)
        plt.close()

        # 保存生成过程可视化（最后一个epoch）
        if epoch == config.num_epochs - 1:
            fig, axes = plt.subplots(1, len(intermediates), figsize=(15, 3))
            for idx, img in enumerate(intermediates):
                img_display = (img[0, 0] + 1) / 2
                axes[idx].imshow(img_display.numpy(), cmap='gray')
                axes[idx].axis('off')
                axes[idx].set_title(f'Step {idx * 100}')
            plt.tight_layout()
            plt.savefig(os.path.join(config.sample_dir, "generation_process.png"))
            plt.close()


def plot_loss_curves(train_losses, mse_losses=None, lpips_losses=None, save_dir="./"):
    """
    绘制损失曲线
    """
    plt.figure(figsize=(12, 6))

    epochs = range(1, len(train_losses) + 1)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Total Loss', linewidth=2)
    if mse_losses:
        plt.plot(epochs, mse_losses, 'r-', label='MSE Loss', linewidth=2)
    if lpips_losses:
        plt.plot(epochs, lpips_losses, 'g-', label='LPIPS Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Curves', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # 添加最佳损失标记
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses, 'b-', label='Total Loss', linewidth=2)
    best_epoch = np.argmin(train_losses) + 1
    best_loss = min(train_losses)
    plt.plot(best_epoch, best_loss, 'go', markersize=10)
    plt.annotate(f'Best: {best_loss:.6f}',
                 xy=(best_epoch, best_loss),
                 xytext=(best_epoch + 2, best_loss),
                 fontsize=9)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Total Loss Curve', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=150)
    plt.close()
    print(f"Loss curve saved to {save_dir}/loss_curve.png")


def save_loss_history(loss_history, save_dir="./"):
    """
    保存损失历史到JSON文件
    """
    loss_path = os.path.join(save_dir, "loss_history.json")
    with open(loss_path, 'w') as f:
        json.dump(loss_history, f, indent=4)
    print(f"Loss history saved to {loss_path}")


def main():
    """主训练函数"""
    # 加载配置
    config = Config()
    print(f"Using device: {config.device}")
    print(f"Data directory: {config.data_dir}")
    print(f"Mask type: {config.mask_type}")
    print(f"Condition block type: {config.cond_block_type}")
    print(f"LPIPS loss: {config.use_lpips}, weight: {config.lpips_loss_weight}")

    # 创建目录
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.sample_dir, exist_ok=True)

    # 准备数据
    print("Loading data...")
    try:
        train_dataloader = prepare_data(shuffle=True)
        sample_dataloader = prepare_data(shuffle=False)
        if len(train_dataloader) == 0:
            print("Error: No data found! Please check the directory structure.")
            print(f"Expected: {config.data_dir}/angiographs/patient/patient_X_mask.png")
            if config.mask_type == "2d":
                print(f"          {config.data_dir}/mask2D/patient_mip.npy")
            else:
                print(f"          {config.data_dir}/mask/patient.nii.gz")
            return
        print(f"Number of batches: {len(train_dataloader)}")
        print(f"Total images: {len(train_dataloader.dataset)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 获取噪声调度
    print("Creating noise schedule...")
    betas = get_noise_schedule(config)

    # 创建扩散过程
    diffusion = GaussianDiffusion(betas, config.device)

    # 创建模型
    print("Creating model...")
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
    ).to(config.device)

    # 打印模型参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f}M")

    # ========== 初始化LPIPS损失函数 ==========
    lpips_loss_fn = None
    if config.use_lpips:
        lpips_loss_fn = lpips.LPIPS(net=config.lpips_net).to(config.device)
        print(f"LPIPS initialized with net={config.lpips_net}")
    # =====================================

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs * len(train_dataloader)
    )

    # 记录损失
    train_losses = []
    mse_losses = []
    lpips_losses = []
    best_loss = float('inf')

    # 训练循环
    print("Starting training...")

    for epoch in range(config.num_epochs):
        avg_total_loss, avg_mse_loss, avg_lpips_loss, batch_losses = train_one_epoch(
            model, diffusion, train_dataloader, optimizer, config, epoch, lpips_loss_fn
        )
        scheduler.step()

        train_losses.append(avg_total_loss)
        mse_losses.append(avg_mse_loss)
        if config.use_lpips:
            lpips_losses.append(avg_lpips_loss)

        print(f"Epoch {epoch + 1}/{config.num_epochs} completed.")
        print(f"  Total Loss: {avg_total_loss:.6f}, MSE: {avg_mse_loss:.6f}, LPIPS: {avg_lpips_loss:.6f}")

        # 保存最佳模型
        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_total_loss,
                'mse_loss': avg_mse_loss,
                'lpips_loss': avg_lpips_loss,
                'config_dict': config.__dict__
            }, os.path.join(config.checkpoint_dir, "best_model.pt"))
            print(f"Saved best model with loss: {avg_total_loss:.6f}")

        # 定期采样
        if (epoch + 1) % config.sample_frequency == 0:
            print(f"\nGenerating samples...")
            sample_and_save(model, diffusion, sample_dataloader, config, epoch)

            # 更新损失曲线
            plot_loss_curves(train_losses, mse_losses, lpips_losses, save_dir=config.sample_dir)
            save_loss_history({
                'train_losses': train_losses,
                'mse_losses': mse_losses,
                'lpips_losses': lpips_losses if config.use_lpips else [],
                'best_loss': best_loss,
                'best_epoch': train_losses.index(best_loss) + 1 if best_loss != float('inf') else 0,
                'num_epochs': epoch + 1,
                'config': {
                    'learning_rate': config.learning_rate,
                    'batch_size': config.batch_size,
                    'cond_block_type': config.cond_block_type,
                    'mask_type': config.mask_type,
                    'use_lpips': config.use_lpips,
                    'lpips_loss_weight': config.lpips_loss_weight
                }
            }, save_dir=config.checkpoint_dir)

    # 训练完成，绘制最终损失曲线
    plot_loss_curves(train_losses, mse_losses, lpips_losses, save_dir=config.sample_dir)
    save_loss_history({
        'train_losses': train_losses,
        'mse_losses': mse_losses,
        'lpips_losses': lpips_losses if config.use_lpips else [],
        'best_loss': best_loss,
        'best_epoch': train_losses.index(best_loss) + 1,
        'num_epochs': config.num_epochs,
        'config': {
            'learning_rate': config.learning_rate,
            'batch_size': config.batch_size,
            'cond_block_type': config.cond_block_type,
            'mask_type': config.mask_type,
            'use_lpips': config.use_lpips,
            'lpips_loss_weight': config.lpips_loss_weight
        }
    }, save_dir=config.checkpoint_dir)

    # 保存最终模型
    torch.save({
        'epoch': config.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_total_loss,
        'mse_loss': avg_mse_loss,
        'lpips_loss': avg_lpips_loss if config.use_lpips else 0,
        'config_dict': config.__dict__
    }, os.path.join(config.checkpoint_dir, "final_model.pt"))

    print("Training completed!")
    print(f"Best loss: {best_loss:.6f} at epoch {train_losses.index(best_loss) + 1}")
    print(f"Loss curve saved to {config.sample_dir}/loss_curve.png")


if __name__ == "__main__":
    import numpy as np

    main()