"""
评估脚本 - 自动配对 TEST 中的真实图像与 Generated 中的生成图像
指标：PSNR, SSIM, FID, KID
"""

import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from PIL import Image
import numpy as np
from tqdm import tqdm
import re

from config import Config


class PairedEvaluationDataset(Dataset):
    """
    自动从 TEST 和 Generated 文件夹中配对图像。
    """
    def __init__(self, test_root, generated_root, sample_index=0, transform=None):
        self.test_root = test_root
        self.generated_root = generated_root
        self.sample_index = sample_index
        self.transform = transform
        self.pairs = []

        angio_dir = os.path.join(test_root, 'angiographs')
        if not os.path.isdir(angio_dir):
            raise NotADirectoryError(f"Angiographs directory not found: {angio_dir}")

        for patient in os.listdir(angio_dir):
            patient_angio_dir = os.path.join(angio_dir, patient)
            if not os.path.isdir(patient_angio_dir):
                continue
            for img_file in os.listdir(patient_angio_dir):
                if not img_file.endswith('.png'):
                    continue
                match = re.match(r'.+?_(\d+)_mask\.png', img_file)
                if not match:
                    print(f"Warning: Cannot parse view from {img_file}, skip")
                    continue
                view = int(match.group(1))
                real_path = os.path.join(patient_angio_dir, img_file)

                fake_dir = os.path.join(generated_root, patient, f"view_{view}")
                if not os.path.isdir(fake_dir):
                    print(f"Warning: Generated folder missing for {patient} view {view}: {fake_dir}")
                    continue
                fake_files = sorted([f for f in os.listdir(fake_dir) if f.startswith('sample_') and f.endswith('.png')])
                if self.sample_index >= len(fake_files):
                    print(f"Warning: sample_index {self.sample_index} out of range for {fake_dir}, skip")
                    continue
                fake_path = os.path.join(fake_dir, fake_files[self.sample_index])
                self.pairs.append((real_path, fake_path))

        print(f"Found {len(self.pairs)} paired images.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        real_path, fake_path = self.pairs[idx]
        real = Image.open(real_path).convert('L')
        fake = Image.open(fake_path).convert('L')
        if self.transform:
            real = self.transform(real)
            fake = self.transform(fake)
        return real, fake


def evaluate(test_root="./TEST", generated_root="./Generated", batch_size=4, device="cuda", output_file="evaluation_results.json"):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = PairedEvaluationDataset(test_root, generated_root, sample_index=0, transform=transform)
    if len(dataset) == 0:
        print("No paired images found. Please check TEST and Generated folders.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 初始化指标
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    fid = FrechetInceptionDistance(feature=2048).to(device)
    kid = KernelInceptionDistance(subset_size=min(50, len(dataset))).to(device)

    psnr_vals = []
    ssim_vals = []

    print(f"Evaluating {len(dataset)} image pairs...")
    for real, fake in tqdm(dataloader):
        real = real.to(device)
        fake = fake.to(device)

        real = real.clamp(0, 1)
        fake = fake.clamp(0, 1)

        psnr_vals.append(psnr(fake, real).item())
        ssim_vals.append(ssim(fake, real).item())

        # FID/KID 需要 RGB 三通道且类型为 uint8 (0-255)
        real_rgb = (real.repeat(1, 3, 1, 1) * 255).to(torch.uint8)
        fake_rgb = (fake.repeat(1, 3, 1, 1) * 255).to(torch.uint8)
        fid.update(real_rgb, real=True)
        fid.update(fake_rgb, real=False)
        kid.update(real_rgb, real=True)
        kid.update(fake_rgb, real=False)

    mean_psnr = np.mean(psnr_vals)
    mean_ssim = np.mean(ssim_vals)
    fid_score = fid.compute().item()
    kid_mean, kid_std = kid.compute()
    kid_mean = kid_mean.item()
    kid_std = kid_std.item()

    results = {
        "PSNR_dB": round(mean_psnr, 4),
        "SSIM": round(mean_ssim, 6),
        "FID": round(fid_score, 4),
        "KID_mean": round(kid_mean, 6),
        "KID_std": round(kid_std, 6),
        "num_samples": len(dataset)
    }

    print("\n====== Evaluation Results ======")
    for k, v in results.items():
        print(f"{k}: {v}")
    print("================================\n")

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    TEST_ROOT = "./TEST"
    GENERATED_ROOT = "./Generated-l&g&s"
    OUTPUT_FILE = "./evaluation_results-l&g&s.json"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    evaluate(TEST_ROOT, GENERATED_ROOT, batch_size=4, device=DEVICE, output_file=OUTPUT_FILE)