"""
采样脚本：使用训练好的模型生成CT图像
"""

import os
import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from config import Config
from utils.noise_schedule import get_noise_schedule
from utils.diffusion import GaussianDiffusion
from models.unet import UNet


def load_model(checkpoint_path, config):
    """加载训练好的模型"""
    model = UNet(
        in_channels=config.channels,
        out_channels=config.channels,
        base_channels=config.base_channels,
        time_emb_dim=config.time_emb_dim
    ).to(config.device)

    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Training loss: {checkpoint.get('loss', 'unknown')}")

    return model


def main():
    """主采样函数"""
    config = Config()

    # 创建输出目录
    os.makedirs("./generated_ct", exist_ok=True)

    # 加载噪声调度
    betas = get_noise_schedule(config)
    diffusion = GaussianDiffusion(betas, config.device)

    # 加载模型
    checkpoint_path = os.path.join(config.checkpoint_dir, "best_model.pt")
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using train.py")
        return

    model = load_model(checkpoint_path, config)

    # 生成样本
    print("Generating CT samples...")
    num_samples = 64
    samples, intermediates = diffusion.sample(
        model,
        config.image_size,
        batch_size=num_samples,
        channels=config.channels,
        progress=True
    )

    # 反归一化并保存
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)

    save_path = "./generated_ct/final_samples.png"
    save_image(samples, save_path, nrow=8)
    print(f"Saved samples to {save_path}")

    # 可视化生成过程
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(10):
        step_idx = i * (len(intermediates) // 10)
        if step_idx >= len(intermediates):
            step_idx = -1
        img = intermediates[step_idx][0, 0]
        img_display = (img + 1) / 2
        axes[i // 5, i % 5].imshow(img_display.numpy(), cmap='gray')
        axes[i // 5, i % 5].axis('off')
        if step_idx == len(intermediates) - 1:
            axes[i // 5, i % 5].set_title('Final')
        else:
            axes[i // 5, i % 5].set_title(f'Step {step_idx * 100}')

    plt.tight_layout()
    plt.savefig("./generated_ct/sampling_process.png")
    plt.show()
    print("Saved sampling process visualization to ./generated_ct/sampling_process.png")


if __name__ == "__main__":
    main()