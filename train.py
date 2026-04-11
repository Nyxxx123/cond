"""
主训练脚本 - CT图像版本
"""

import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import Config
from utils.noise_schedule import get_noise_schedule
from utils.diffusion import GaussianDiffusion
from utils.dataset import create_ct_dataloader  # 改用新的数据集
from models.unet import UNet

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def prepare_data(config):
    """准备CT数据集"""
    dataloader = create_ct_dataloader(config, shuffle=True)
    return dataloader


def train_one_epoch(model, diffusion, dataloader, optimizer, config, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")
    for batch_idx, (images, _) in enumerate(pbar):
        images = images.to(config.device)
        batch_size = images.shape[0]

        # 随机采样时间步
        t = torch.randint(0, config.timesteps, (batch_size,), device=config.device).long()

        # 计算损失
        loss = diffusion.p_losses(model, images, t)

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
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{avg_loss:.4f}'
        })

    return total_loss / num_batches


def sample_and_save(model, diffusion, config, epoch):
    """生成样本并保存"""
    model.eval()
    with torch.no_grad():
        samples, intermediates = diffusion.sample(
            model,
            config.image_size,
            batch_size=config.sample_batch_size,
            channels=config.channels,
            sampler_type=config.sampler_type,
            ddim_steps=config.ddim_steps,
            eta=config.ddim_eta,
            progress=True
        )

        # 反归一化到[0,1]并保存
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)

        save_path = os.path.join(config.sample_dir, f"epoch_{epoch + 1}.png")
        save_image(samples, save_path, nrow=4)

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


def main():
    """主训练函数"""
    # 加载配置
    config = Config()
    print(f"Using device: {config.device}")
    print(f"Data directory: {config.data_dir}")

    # 创建目录
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.sample_dir, exist_ok=True)

    # 准备数据
    print("Loading CT data...")
    try:
        dataloader = prepare_data(config)
        if len(dataloader) == 0:
            print("Error: No data found! Please check the PARSE directory.")
            print(f"Expected directory: {os.path.abspath(config.data_dir)}")
            return
        print(f"Number of batches: {len(dataloader)}")
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
    model = UNet(
        in_channels=config.channels,
        out_channels=config.channels,
        base_channels=config.base_channels,
        time_emb_dim=config.time_emb_dim
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
        T_max=config.num_epochs * len(dataloader)
    )

    # 训练循环
    print("Starting training...")
    best_loss = float('inf')

    for epoch in range(config.num_epochs):
        avg_loss = train_one_epoch(
            model, diffusion, dataloader, optimizer, config, epoch
        )
        scheduler.step()

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
            sample_and_save(model, diffusion, config, epoch)

    print("Training completed!")

    # 保存最终模型
    torch.save({
        'epoch': config.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'config': config
    }, os.path.join(config.checkpoint_dir, "final_model.pt"))


if __name__ == "__main__":
    main()