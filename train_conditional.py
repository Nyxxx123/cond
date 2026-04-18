"""
条件扩散模型训练脚本 - 肺动脉造影版本
使用血管掩码作为条件（支持2D MIP或3D原始掩码）
"""

import os
import json
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt

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


def train_one_epoch(model, diffusion, dataloader, optimizer, config, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    batch_losses = []  # 记录每个batch的loss

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")
    for batch_idx, batch in enumerate(pbar):
        # 获取数据（条件数据集返回字典）
        targets = batch['target'].to(config.device)  # 造影图像 [B,1,H,W]
        masks = batch['mask'].to(config.device)      # 掩码（2D或3D）
        batch_size = targets.shape[0]

        # 随机采样时间步
        t = torch.randint(0, config.timesteps, (batch_size,), device=config.device).long()

        # 计算损失（传入条件）
        loss = diffusion.p_losses(model, targets, t, cond=masks)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        optimizer.step()

        # 更新进度条
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        batch_losses.append(loss.item())
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{avg_loss:.4f}'
        })

    return total_loss / num_batches, batch_losses


def sample_and_save(model, diffusion, dataloader, config, epoch):
    """生成样本并保存"""
    model.eval()

    # 从dataloader中获取一个batch的掩码作为条件
    for batch in dataloader:
        test_masks = batch['mask'][:config.sample_batch_size].to(config.device)
        break

    # 关键修改：获取实际batch大小
    actual_batch_size = test_masks.shape[0]
    num_samples = actual_batch_size  # 使用实际数量，而不是config.sample_batch_size

    with torch.no_grad():
        samples, intermediates = diffusion.sample(
            model,
            config.image_size,
            batch_size=num_samples,  # 使用实际数量
            channels=config.channels,
            sampler_type=config.sampler_type,
            ddim_steps=config.ddim_steps,
            eta=config.ddim_eta,
            cond=test_masks,
            progress=True
        )

        # 反归一化到[0,1]并保存
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)

        save_path = os.path.join(config.sample_dir, f"epoch_{epoch + 1}.png")
        save_image(samples, save_path, nrow=4)

        # 保存条件掩码和生成结果的对比图
        # 关键修改：使用实际数量，最多显示8个
        display_num = min(8, num_samples)
        fig, axes = plt.subplots(2, display_num,
                                 figsize=(2 * display_num, 4))
        if display_num == 1:
            axes = axes.reshape(-1, 1)

        for i in range(display_num):
            # 第一行：条件掩码（支持2D和3D）
            mask_disp = test_masks[i, 0].cpu().numpy()
            # 如果是3D掩码，取中间切片显示
            if mask_disp.ndim == 3:
                mid_slice = mask_disp.shape[0] // 2
                mask_disp = mask_disp[mid_slice]
            axes[0, i].imshow(mask_disp, cmap='hot')
            axes[0, i].set_title("Condition", fontsize=8)
            axes[0, i].axis('off')

            # 第二行：生成结果
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


def plot_loss_curves(train_losses, val_losses=None, save_dir="./"):
    """
    绘制损失曲线

    Args:
        train_losses: 每个epoch的训练损失列表
        val_losses: 每个epoch的验证损失列表（可选）
        save_dir: 保存目录
    """
    plt.figure(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)

    if val_losses:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Curve', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # 添加最佳损失标记
    best_epoch = np.argmin(train_losses) + 1
    best_loss = min(train_losses)
    plt.plot(best_epoch, best_loss, 'go', markersize=10, label=f'Best: {best_loss:.6f}')
    plt.annotate(f'Best: {best_loss:.6f}',
                 xy=(best_epoch, best_loss),
                 xytext=(best_epoch + 2, best_loss),
                 fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=150)
    plt.close()

    print(f"Loss curve saved to {save_dir}/loss_curve.png")


def save_loss_history(loss_history, save_dir="./"):
    """
    保存损失历史到JSON文件

    Args:
        loss_history: 包含训练损失的字典
        save_dir: 保存目录
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
    print(f"Mask type: {config.mask_type}")  # 新增：打印掩码类型
    print(f"Condition block type: {config.cond_block_type}")

    # 创建目录
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.sample_dir, exist_ok=True)

    # 准备数据
    print("Loading data...")
    try:
        train_dataloader = prepare_data(shuffle=True)
        sample_dataloader = prepare_data(shuffle=False)  # 用于采样可视化
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

    # 创建模型（修改点1：添加 mask_type 参数）
    print("Creating model...")
    model = ConditionalUNet(
        in_channels=config.channels,
        out_channels=config.channels,
        base_channels=config.base_channels,
        cond_dim=config.cond_dim,
        time_emb_dim=config.time_emb_dim,
        block_type=config.cond_block_type,
        mask_type=config.mask_type  # 新增这一行
    ).to(config.device)

    # 打印模型参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f}M")

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
    best_loss = float('inf')

    # 训练循环
    print("Starting training...")

    for epoch in range(config.num_epochs):
        avg_loss, batch_losses = train_one_epoch(
            model, diffusion, train_dataloader, optimizer, config, epoch
        )
        scheduler.step()

        train_losses.append(avg_loss)

        print(f"Epoch {epoch + 1}/{config.num_epochs} completed. Average loss: {avg_loss:.6f}")

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config
            }, os.path.join(config.checkpoint_dir, "best_model.pt"))
            print(f"Saved best model with loss: {avg_loss:.6f}")

        # 定期采样
        if (epoch + 1) % config.sample_frequency == 0:
            sample_and_save(model, diffusion, sample_dataloader, config, epoch)

            # 每次采样时也更新损失曲线
            plot_loss_curves(train_losses, save_dir=config.sample_dir)
            save_loss_history({
                'train_losses': train_losses,
                'best_loss': best_loss,
                'best_epoch': train_losses.index(best_loss) + 1 if best_loss != float('inf') else 0,
                'num_epochs': epoch + 1,
                'config': {
                    'learning_rate': config.learning_rate,
                    'batch_size': config.batch_size,
                    'cond_block_type': config.cond_block_type,
                    'mask_type': config.mask_type  # 新增
                }
            }, save_dir=config.checkpoint_dir)

    # 训练完成，绘制最终损失曲线
    plot_loss_curves(train_losses, save_dir=config.sample_dir)
    save_loss_history({
        'train_losses': train_losses,
        'best_loss': best_loss,
        'best_epoch': train_losses.index(best_loss) + 1,
        'num_epochs': config.num_epochs,
        'config': {
            'learning_rate': config.learning_rate,
            'batch_size': config.batch_size,
            'cond_block_type': config.cond_block_type,
            'mask_type': config.mask_type  # 新增
        }
    }, save_dir=config.checkpoint_dir)

    # 保存最终模型
    torch.save({
        'epoch': config.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'config': config
    }, os.path.join(config.checkpoint_dir, "final_model.pt"))

    print("Training completed!")
    print(f"Best loss: {best_loss:.6f} at epoch {train_losses.index(best_loss) + 1}")
    print(f"Loss curve saved to {config.sample_dir}/loss_curve.png")


if __name__ == "__main__":
    import numpy as np
    main()