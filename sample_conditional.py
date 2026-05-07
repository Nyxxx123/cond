"""
条件采样脚本 - 自动批量处理 TEST 文件夹
TEST 文件夹结构：
    TEST/
    ├── mask/                    # 3D 掩码 (.nii.gz)
    │   ├── patient001.nii.gz
    │   └── patient002.nii.gz
    ├── angiographs/             # 真实有造影图像（用于对比）
    │   ├── patient001/
    │   │   ├── patient001_0_mask.png
    │   │   └── patient001_1_mask.png
    │   └── patient002/
    │       ├── patient002_0_mask.png
    │       └── ...
    └── non_angiographs/         # 无造影CT（可选）
        ├── patient001/
        │   ├── patient001_0_mask.png
        │   └── patient001_1_mask.png
        └── patient002/
            ├── patient002_0_mask.png
            └── ...
"""

import os
import torch
import numpy as np
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import nibabel as nib
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

from config import Config
from utils.noise_schedule import get_noise_schedule
from utils.diffusion import GaussianDiffusion
from models.cond_unet import ConditionalUNet
from utils.analyse_angle import get_angle_info

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def load_model(checkpoint_path, config, device):
    """加载训练好的模型（优先 EMA 版本）"""
    ema_path = checkpoint_path.replace('.pt', '_ema.pt')
    if os.path.exists(ema_path):
        print(f"Loading EMA model from {ema_path}")
        checkpoint = torch.load(ema_path, map_location=device, weights_only=False)
    else:
        print(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = ConditionalUNet(
        in_channels=config.channels,
        out_channels=config.channels,
        base_channels=config.base_channels,
        cond_dim=config.cond_dim,
        time_emb_dim=config.time_emb_dim,
        block_type=config.cond_block_type,
        mask_type="3d",
        use_angle=config.use_angle_condition,
        angle_dim=config.angle_dim,
        use_non_angio=config.use_non_angio
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model epoch {checkpoint.get('epoch', 'unknown')}, loss {checkpoint.get('loss', 'unknown'):.6f}")
    return model


def load_3d_mask(mask_path, target_size, device):
    """加载3D掩码并下采样"""
    nii = nib.load(mask_path)
    mask_3d = nii.get_fdata()[np.newaxis, ...]
    mask_tensor = torch.from_numpy(mask_3d).float()
    if mask_tensor.shape[1:] != target_size:
        mask_tensor = mask_tensor.unsqueeze(0)
        mask_tensor = torch.nn.functional.interpolate(
            mask_tensor, size=target_size, mode='trilinear', align_corners=False
        )
        mask_tensor = mask_tensor.squeeze(0)
    if mask_tensor.max() > 0:
        mask_tensor = mask_tensor / mask_tensor.max()
    return mask_tensor.to(device)


def load_image(image_path, target_size, device, to_range="01"):
    """加载任意2D图像"""
    img = Image.open(image_path).convert('L')
    img = img.resize(target_size, Image.BILINEAR)
    img_tensor = transforms.ToTensor()(img)
    if to_range == "minus1_1":
        img_tensor = img_tensor * 2 - 1
    return img_tensor.unsqueeze(0).to(device)


def get_angle_from_filename(filename, angle_rep):
    """从文件名解析角度向量"""
    info = get_angle_info(os.path.basename(filename), angle_rep=angle_rep)
    return torch.from_numpy(info['angle_vector']).float()


@torch.no_grad()
def sample_conditional(model, diffusion, mask_tensor, angle_tensor, non_angio_tensor,
                       config, device, num_samples):
    """执行条件采样，生成 num_samples 个样本"""
    mask_batch = mask_tensor.repeat(num_samples, 1, 1, 1, 1)
    angle_batch = angle_tensor.repeat(num_samples, 1) if angle_tensor is not None else None
    non_angio_batch = non_angio_tensor.repeat(num_samples, 1, 1, 1) if non_angio_tensor is not None else None

    samples, _ = diffusion.sample(
        model,
        config.image_size,
        batch_size=num_samples,
        channels=config.channels,
        sampler_type=config.sampler_type,
        ddim_steps=config.ddim_steps,
        eta=config.ddim_eta,
        mask=mask_batch,
        angle=angle_batch,
        non_angio=non_angio_batch,
        progress=False
    )
    # 反归一化到 [0,1]
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)
    return samples


def scan_test_folder(test_root):
    """
    扫描 TEST 文件夹，返回所有待测试的样本信息列表。
    每个样本为一个字典: {
        'patient': 患者ID,
        'view': 视角编号,
        'mask_path': 3D掩码路径,
        'gt_path': 真实造影图像路径,
        'non_angio_path': 无造影CT路径（可能为 None）
    }
    """
    mask_dir = os.path.join(test_root, 'mask')
    angio_dir = os.path.join(test_root, 'angiographs')
    non_angio_dir = os.path.join(test_root, 'non_angiographs') if os.path.exists(os.path.join(test_root, 'non_angiographs')) else None

    samples = []

    # 获取所有掩码文件（.nii.gz 或 .nii）
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.nii.gz', '.nii'))]
    for mask_file in mask_files:
        # 患者ID = 文件名去除扩展名
        patient_id = mask_file.replace('.nii.gz', '').replace('.nii', '')
        mask_path = os.path.join(mask_dir, mask_file)

        # 对应的 angiographs 文件夹
        patient_angio_dir = os.path.join(angio_dir, patient_id)
        if not os.path.isdir(patient_angio_dir):
            print(f"Warning: No angiographs folder for patient {patient_id}, skip")
            continue

        # 遍历该患者的所有造影图像（假设命名格式为 patient_id_{view}_mask.png）
        angio_files = [f for f in os.listdir(patient_angio_dir) if f.endswith('.png')]
        for angio_file in angio_files:
            # 提取视角编号
            parts = angio_file.replace('.png', '').split('_')
            if len(parts) < 2:
                continue
            try:
                view = int(parts[-2])   # 倒数第二个是视角编号，如 patient001_0_mask -> 0
            except:
                continue
            gt_path = os.path.join(patient_angio_dir, angio_file)

            # 无造影CT路径（如果存在）
            non_angio_path = None
            if non_angio_dir is not None:
                candidate = os.path.join(non_angio_dir, patient_id, angio_file)
                if os.path.exists(candidate):
                    non_angio_path = candidate

            samples.append({
                'patient': patient_id,
                'view': view,
                'mask_path': mask_path,
                'gt_path': gt_path,
                'non_angio_path': non_angio_path
            })
    return samples


def main():
    # ==================== 配置参数 ====================
    TEST_ROOT = "./TEST"                # TEST 文件夹根目录
    OUTPUT_BASE = "./Generated-l&g&s"         # 输出根目录（内部自动按患者/视角组织）
    CHECKPOINT_NAME = "best_model.pt"   # 检查点文件名
    NUM_SAMPLES = 4                     # 每个条件生成几张图像
    # =================================================

    config = Config()
    device = config.device

    print("="*60)
    print("批量采样 - 自动处理 TEST 文件夹")
    print(f"Test root: {TEST_ROOT}")
    print(f"Output base: {OUTPUT_BASE}")
    print(f"Device: {device}")
    print("="*60)

    # 检查 TEST 文件夹是否存在
    if not os.path.isdir(TEST_ROOT):
        raise NotADirectoryError(f"TEST folder not found: {TEST_ROOT}")

    # 加载模型（仅一次）
    checkpoint_full = os.path.join(config.checkpoint_dir, CHECKPOINT_NAME)
    model = load_model(checkpoint_full, config, device)

    # 扩散过程
    betas = get_noise_schedule(config)
    diffusion = GaussianDiffusion(betas, device, prediction_type=config.prediction_type)

    # 获取所有测试样本
    samples = scan_test_folder(TEST_ROOT)
    print(f"Found {len(samples)} test samples (patient-view pairs).")

    if not samples:
        print("No samples found. Please check TEST folder structure.")
        return

    # 逐个处理样本
    for sample in tqdm(samples, desc="Processing samples"):
        patient = sample['patient']
        view = sample['view']
        mask_path = sample['mask_path']
        gt_path = sample['gt_path']
        non_angio_path = sample['non_angio_path']

        # 创建该样本的输出子目录
        output_dir = os.path.join(OUTPUT_BASE, patient, f"view_{view}")
        os.makedirs(output_dir, exist_ok=True)

        # 加载条件
        mask = load_3d_mask(mask_path, config.mask_3d_size, device)
        if config.use_angle_condition:
            angle = get_angle_from_filename(gt_path, config.angle_rep)
            angle = angle.unsqueeze(0).to(device)
        else:
            angle = None
        non_angio = None
        if config.use_non_angio and non_angio_path is not None:
            non_angio = load_image(non_angio_path, (config.image_size, config.image_size), device, to_range="minus1_1")
        elif config.use_non_angio:
            print(f"Warning: No non-angio image for {patient} view {view}, using zeros.")
            non_angio = torch.zeros(1, 1, config.image_size, config.image_size).to(device)

        # 真实图像（用于对比）
        try:
            gt_image = load_image(gt_path, (config.image_size, config.image_size), device, to_range="01")
        except Exception as e:
            print(f"Failed to load GT image {gt_path}: {e}, skip.")
            continue

        # 采样生成
        samples_gen = sample_conditional(model, diffusion, mask, angle, non_angio,
                                         config, device, NUM_SAMPLES)

        # 保存生成图像
        for i, s in enumerate(samples_gen):
            save_image(s, os.path.join(output_dir, f"sample_{i:02d}.png"))
        save_image(samples_gen, os.path.join(output_dir, "grid.png"), nrow=2)

        # 生成对比图（GT vs 第一个生成样本）
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        gt_disp = gt_image[0, 0].cpu().numpy()
        axes[0].imshow(gt_disp, cmap='gray')
        axes[0].set_title("Ground Truth")
        axes[0].axis('off')
        axes[1].imshow(samples_gen[0, 0].cpu().numpy(), cmap='gray')
        axes[1].set_title("Generated")
        axes[1].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "comparison.png"), dpi=150)
        plt.close()

    print(f"\nBatch sampling completed. Results saved to {OUTPUT_BASE}")


if __name__ == "__main__":
    main()