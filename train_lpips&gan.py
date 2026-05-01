"""
条件扩散模型训练脚本 - 肺动脉造影版本
支持: 纯LPIPS / LPIPS+GAN (通过 config.use_gan 切换)
当 use_gan=False 时，行为与 train_conditional_lpips.py 完全一致
"""

import os
import json
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
import lpips

from config import Config
from utils.noise_schedule import get_noise_schedule
from utils.diffusion import GaussianDiffusion
from utils.dataset import create_dataloader
from models.cond_unet import ConditionalUNet

# 只在启用GAN时导入判别器
if Config().use_gan:
    from models.discriminator import Discriminator

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def prepare_data(shuffle=True):
    """准备条件数据集"""
    dataloader = create_dataloader(shuffle=shuffle)
    return dataloader


def sample_and_save(model, diffusion, dataloader, config, epoch):
    """生成样本并保存"""
    model.eval()

    for batch in dataloader:
        test_masks = batch['mask'][:config.sample_batch_size].to(config.device)
        test_angles = batch['angle'][:config.sample_batch_size].to(config.device)
        test_targets = batch['target'][:config.sample_batch_size]
        # ========== 新增：获取无造影CT条件 ==========
        if config.use_non_angio:
            test_non_angio = batch['non_angio'][:config.sample_batch_size].to(config.device)
        else:
            test_non_angio = None
        # ========================================
        break

    actual_batch_size = test_masks.shape[0]
    num_samples = actual_batch_size

    with torch.no_grad():
        # ========== 修改：传递 non_angio 参数 ==========
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
            non_angio=test_non_angio,   # 新增
            progress=True
        )
        # ============================================

        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)
        targets_disp = (test_targets + 1) / 2
        targets_disp = torch.clamp(targets_disp, 0, 1)

        save_path = os.path.join(config.sample_dir, f"epoch_{epoch + 1}.png")
        save_image(samples, save_path, nrow=4)

        display_num = min(8, num_samples)
        fig, axes = plt.subplots(2, display_num, figsize=(2 * display_num, 4))
        if display_num == 1:
            axes = axes.reshape(-1, 1)

        for i in range(display_num):
            axes[0, i].imshow(targets_disp[i, 0].cpu().numpy(), cmap='gray')
            axes[0, i].set_title(f"Ground Truth", fontsize=8)
            axes[0, i].axis('off')
            axes[1, i].imshow(samples[i, 0].cpu().numpy(), cmap='gray')
            axes[1, i].set_title(f"Generated", fontsize=8)
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(config.sample_dir, f"epoch_{epoch + 1}_comparison.png"), dpi=150)
        plt.close()

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


def plot_loss_curves(train_losses, mse_losses, lpips_losses, gan_losses, disc_losses, save_dir="./"):
    """绘制损失曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = range(1, len(train_losses) + 1)

    axes[0, 0].plot(epochs, train_losses, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Generator Loss')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, mse_losses, 'r-', label='MSE', linewidth=2)
    if lpips_losses:
        axes[0, 1].plot(epochs, lpips_losses, 'g-', label='LPIPS', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('MSE & LPIPS Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    if gan_losses:
        axes[1, 0].plot(epochs, gan_losses, 'm-', label='GAN Loss', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('GAN Generator Loss')
        axes[1, 0].grid(True, alpha=0.3)

    if disc_losses:
        axes[1, 1].plot(epochs, disc_losses, 'c-', label='Discriminator Loss', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Discriminator Loss')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curves.png"), dpi=150)
    plt.close()


def save_loss_history(loss_history, save_dir="./"):
    loss_path = os.path.join(save_dir, "loss_history.json")
    with open(loss_path, 'w') as f:
        json.dump(loss_history, f, indent=4)


def train_one_epoch(model, diffusion, dataloader, optimizer_G, config, epoch, lpips_loss_fn,
                    discriminator=None, optimizer_D=None):
    """训练一个epoch - 支持GAN推迟启动"""
    model.train()
    if discriminator is not None:
        discriminator.train()

    total_loss_G = 0
    total_mse_loss = 0
    total_lpips_loss = 0
    total_gan_loss = 0
    total_loss_D = 0
    num_batches = len(dataloader)

    # ========== 判断是否启用GAN（推迟启动） ==========
    use_gan = (discriminator is not None and epoch >= config.gan_start_epoch)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")
    for batch_idx, batch in enumerate(pbar):
        targets = batch['target'].to(config.device)
        masks = batch['mask'].to(config.device)
        angles = batch['angle'].to(config.device)
        # ========== 新增：获取无造影CT条件 ==========
        if config.use_non_angio:
            non_angio = batch['non_angio'].to(config.device)
        else:
            non_angio = None
        # ========================================
        batch_size = targets.shape[0]

        t = torch.randint(0, config.timesteps, (batch_size,), device=config.device).long()
        noise = torch.randn_like(targets)
        x_noisy = diffusion.q_sample(targets, t, noise)

        # ========== 生成器前向（传递 non_angio） ==========
        model_output = model(x_noisy, masks, angles, non_angio=non_angio, t=t)
        # =============================================

        # ========== 根据 prediction_type 计算正确的 target ==========
        if config.prediction_type == "epsilon":
            target = noise
        else:  # v-prediction
            sqrt_alphas_cumprod_t = diffusion._extract(diffusion.sqrt_alphas_cumprod, t, targets.shape)
            sqrt_one_minus_alphas_cumprod_t = diffusion._extract(diffusion.sqrt_one_minus_alphas_cumprod, t, targets.shape)
            target = sqrt_alphas_cumprod_t * noise - sqrt_one_minus_alphas_cumprod_t * targets

        # 计算 MSE 损失
        mse_loss = F.mse_loss(model_output, target)
        # ==================================================

        # ========== 计算 x0_pred ==========
        if config.prediction_type == "epsilon":
            x0_pred = diffusion.predict_x0_from_noise(x_noisy, model_output, t)
        else:  # v-prediction
            x0_pred = diffusion.predict_x0_from_v(x_noisy, model_output, t)
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        # ========== LPIPS损失 ==========
        if config.use_lpips:
            lpips_loss_val = lpips_loss_fn(x0_pred, targets).mean()
            total_lpips_loss += lpips_loss_val.item()
        else:
            lpips_loss_val = torch.tensor(0.0, device=config.device)

        # ========== 生成器总损失 ==========
        loss_G = mse_loss
        if config.use_lpips:
            loss_G = loss_G + config.lpips_loss_weight * lpips_loss_val

        if use_gan:
            fake_pred = discriminator(x0_pred)
            gen_loss = F.mse_loss(fake_pred, torch.ones_like(fake_pred))
            loss_G = loss_G + config.gan_loss_weight * gen_loss
            total_gan_loss += gen_loss.item()

        # 更新生成器
        optimizer_G.zero_grad()
        loss_G.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer_G.step()

        # ========== 判别器前向（独立） ==========
        if use_gan:
            with torch.no_grad():
                # 重新前向传播（传递 non_angio）
                model_output_disc = model(x_noisy, masks, angles, non_angio=non_angio, t=t)
                if config.prediction_type == "epsilon":
                    x0_pred_disc = diffusion.predict_x0_from_noise(x_noisy, model_output_disc, t)
                else:
                    x0_pred_disc = diffusion.predict_x0_from_v(x_noisy, model_output_disc, t)
                    x0_pred_disc = torch.clamp(x0_pred_disc, -1.0, 1.0)

            real_pred = discriminator(targets)
            fake_pred = discriminator(x0_pred_disc)

            real_loss = F.mse_loss(real_pred, torch.ones_like(real_pred))
            fake_loss = F.mse_loss(fake_pred, torch.zeros_like(fake_pred))
            disc_loss = (real_loss + fake_loss) / 2

            total_loss_D += disc_loss.item()

            optimizer_D.zero_grad()
            disc_loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), config.grad_clip)
            optimizer_D.step()
        # =================================

        total_loss_G += loss_G.item()
        total_mse_loss += mse_loss.item()

        avg_loss = total_loss_G / (batch_idx + 1)
        postfix = {'loss': f'{loss_G.item():.4f}', 'mse': f'{mse_loss.item():.4f}', 'avg': f'{avg_loss:.4f}'}
        if config.use_lpips:
            postfix['lpips'] = f'{lpips_loss_val.item():.4f}'
        if use_gan:
            postfix['gan'] = f'{gen_loss.item():.4f}'
            postfix['disc'] = f'{disc_loss.item():.4f}'
        pbar.set_postfix(postfix)

    avg_loss_G = total_loss_G / num_batches
    avg_mse = total_mse_loss / num_batches
    avg_lpips = total_lpips_loss / num_batches if config.use_lpips else 0
    avg_gan = total_gan_loss / num_batches if use_gan else 0
    avg_loss_D = total_loss_D / num_batches if use_gan else 0

    return avg_loss_G, avg_mse, avg_lpips, avg_gan, avg_loss_D


def main():
    config = Config()

    print("=" * 60)
    print("条件扩散模型训练")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Mask type: {config.mask_type}")
    print(f"LPIPS: {config.use_lpips}, weight={config.lpips_loss_weight}")
    print(f"GAN: {config.use_gan}")
    if config.use_gan:
        print(f"  GAN weight: {config.gan_loss_weight}")
        print(f"  Discriminator LR: {config.disc_lr}")
    print("=" * 60)

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.sample_dir, exist_ok=True)

    # 准备数据
    print("Loading data...")
    train_dataloader = prepare_data(shuffle=True)
    sample_dataloader = prepare_data(shuffle=False)

    if len(train_dataloader) == 0:
        print("Error: No data found!")
        return
    print(f"Number of batches: {len(train_dataloader)}")
    print(f"Total images: {len(train_dataloader.dataset)}")

    # 噪声调度
    print("Creating noise schedule...")
    betas = get_noise_schedule(config)
    diffusion = GaussianDiffusion(betas, config.device, prediction_type=config.prediction_type)

    # 创建生成器
    print("Creating generator (UNet)...")
    model = ConditionalUNet(
        in_channels=config.channels,
        out_channels=config.channels,
        base_channels=config.base_channels,
        cond_dim=config.cond_dim,
        time_emb_dim=config.time_emb_dim,
        block_type=config.cond_block_type,
        mask_type=config.mask_type,
        use_angle=config.use_angle_condition,
        angle_dim=config.angle_dim,
        use_non_angio=config.use_non_angio        # 新增参数
    ).to(config.device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Generator parameters: {num_params / 1e6:.2f}M")

    # ========== 根据配置创建判别器 ==========
    discriminator = None
    optimizer_D = None

    if config.use_gan:
        print("Creating discriminator...")
        discriminator = Discriminator(in_channels=config.channels).to(config.device)
        disc_params = sum(p.numel() for p in discriminator.parameters())
        print(f"Discriminator parameters: {disc_params / 1e6:.2f}M")
        optimizer_D = torch.optim.AdamW(
            discriminator.parameters(),
            lr=config.disc_lr,
            weight_decay=config.weight_decay
        )
    # =====================================

    # LPIPS损失
    lpips_loss_fn = lpips.LPIPS(net=config.lpips_net).to(config.device)
    print(f"LPIPS initialized with net={config.lpips_net}")

    # 生成器优化器
    optimizer_G = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_G,
        T_max=config.num_epochs * len(train_dataloader)
    )

    # 记录损失
    train_losses = []
    mse_losses = []
    lpips_losses = []
    gan_losses = []
    disc_losses = []
    best_loss = float('inf')

    print("Starting training...")

    for epoch in range(config.num_epochs):
        # 训练一个epoch
        avg_loss_G, avg_mse, avg_lpips, avg_gan, avg_loss_D = train_one_epoch(
            model, diffusion, train_dataloader, optimizer_G, config, epoch,
            lpips_loss_fn, discriminator, optimizer_D
        )
        scheduler.step()

        train_losses.append(avg_loss_G)
        mse_losses.append(avg_mse)
        lpips_losses.append(avg_lpips)
        if config.use_gan:
            gan_losses.append(avg_gan)
            disc_losses.append(avg_loss_D)

        print(f"\nEpoch {epoch + 1}/{config.num_epochs} completed.")
        print(f"  Generator Loss: {avg_loss_G:.6f}")
        print(f"    MSE: {avg_mse:.6f}, LPIPS: {avg_lpips:.6f}")
        if config.use_gan:
            print(f"    GAN: {avg_gan:.6f}, Discriminator: {avg_loss_D:.6f}")

        # 保存最佳模型
        if avg_loss_G < best_loss:
            best_loss = avg_loss_G
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer_G.state_dict(),
                'loss': avg_loss_G,
                'config_dict': config.__dict__
            }, os.path.join(config.checkpoint_dir, "best_model.pt"))

            if config.use_gan and discriminator is not None:
                torch.save({
                    'discriminator_state_dict': discriminator.state_dict(),
                }, os.path.join(config.checkpoint_dir, "best_discriminator.pt"))

            print(f"  Saved best model with loss: {avg_loss_G:.6f}")

        # 定期采样
        if (epoch + 1) % config.sample_frequency == 0:
            print(f"\nGenerating samples...")
            sample_and_save(model, diffusion, sample_dataloader, config, epoch)

            # 更新损失曲线
            plot_loss_curves(train_losses, mse_losses, lpips_losses, gan_losses, disc_losses,
                            save_dir=config.sample_dir)
            save_loss_history({
                'train_losses': train_losses,
                'mse_losses': mse_losses,
                'lpips_losses': lpips_losses,
                'gan_losses': gan_losses,
                'disc_losses': disc_losses,
                'best_loss': best_loss,
                'best_epoch': train_losses.index(best_loss) + 1,
                'num_epochs': epoch + 1,
                'use_gan': config.use_gan
            }, save_dir=config.checkpoint_dir)

    # 保存最终模型
    torch.save({
        'epoch': config.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer_G.state_dict(),
        'loss': avg_loss_G,
        'config_dict': config.__dict__
    }, os.path.join(config.checkpoint_dir, "final_model.pt"))

    print("Training completed!")
    print(f"Best loss: {best_loss:.6f}")


if __name__ == "__main__":
    import numpy as np
    main()