"""
条件数据集：肺动脉造影 + 2D血管掩码MIP（预计算.npy版本）
数据组织方式：
    data/
    ├── angiographs/                    # 肺动脉造影图像（目标）
    │   ├── patient001/
    │   │   ├── patient001_0_mask.png
    │   │   └── ...
    │   └── patient002/
    ├── mask/                           # 原始3D血管掩码（可选，保留）
    │   ├── patient001.nii.gz
    │   └── patient002.nii.gz
    └── mask2D/                         # 预计算的2D MIP掩码（.npy格式）
        ├── patient001_mip.npy
        └── patient002_mip.npy
"""

import os
import re
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from config import Config
from utils.analyse_angle import get_angle_info


class PulmonaryAngiographyDataset(Dataset):
    """
    肺动脉造影数据集（使用预计算的.npy MIP掩码，保留float32精度）
    """

    def __init__(self, root_dir, image_size=256, transform=None, device='cpu'):
        """
        Args:
            root_dir: 数据根目录
            image_size: 输出2D图像尺寸
            transform: 图像变换
            device: 设备
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.device = device

        # 目录路径
        self.angiographs_dir = os.path.join(root_dir, 'angiographs')
        self.mask_mip_dir = os.path.join(root_dir, 'mask2D')  # 使用预计算的.npy MIP

        # 存储所有图像路径
        self.image_list = []

        # 收集数据
        self._collect_data()

        print(f"找到 {len(self.image_list)} 张造影图像")

        # 定义图像变换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])  # 归一化到[-1,1]
            ])
        else:
            self.transform = transform

    def _extract_info_from_filename(self, filename):
        """
        从文件名提取患者ID和视角序号
        例如: patient001_0_mask.png -> ("patient001", 0)
        """
        pattern = r'(.+)_(\d+)_mask\.png$'
        match = re.match(pattern, filename)

        if match:
            patient_id = match.group(1)
            view_index = int(match.group(2))
            return patient_id, view_index

        raise ValueError(f"无法解析文件名: {filename}，期望格式: patient001_0_mask.png")

    def _collect_data(self):
        """收集所有造影图像和对应的MIP掩码路径"""
        if not os.path.exists(self.angiographs_dir):
            raise FileNotFoundError(f"目录不存在: {self.angiographs_dir}")

        if not os.path.exists(self.mask_mip_dir):
            raise FileNotFoundError(f"MIP目录不存在: {self.mask_mip_dir}\n请先运行 preprocess_masks.py")

        # 遍历患者文件夹
        patient_folders = sorted(os.listdir(self.angiographs_dir))

        for patient_folder in patient_folders:
            patient_angio_dir = os.path.join(self.angiographs_dir, patient_folder)

            if not os.path.isdir(patient_angio_dir):
                continue

            # 查找对应的预计算MIP文件（.npy格式）
            mip_path = os.path.join(self.mask_mip_dir, f"{patient_folder}_mip.npy")
            if not os.path.exists(mip_path):
                print(f"警告: 患者 {patient_folder} 的MIP文件不存在，跳过")
                continue

            # 收集该患者的所有造影图像
            image_files = sorted(os.listdir(patient_angio_dir))
            supported_ext = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}

            for img_file in image_files:
                ext = os.path.splitext(img_file)[1].lower()
                if ext in supported_ext:
                    try:
                        patient_id, view_index = self._extract_info_from_filename(img_file)

                        if patient_id != patient_folder:
                            print(f"警告: 文件名中的患者ID({patient_id})与文件夹名({patient_folder})不一致，跳过")
                            continue

                        self.image_list.append({
                            'image_path': os.path.join(patient_angio_dir, img_file),
                            'mip_path': mip_path,
                            'patient': patient_folder,
                            'img_name': img_file,
                            'view_index': view_index
                        })
                    except ValueError as e:
                        print(f"警告: 跳过文件 {img_file}，{e}")

    def _load_mip_mask(self, mip_path):
        """
        加载预计算的MIP掩码（.npy格式）并转换为张量
        保留原始float32精度
        """
        # 加载.npy文件
        mip_array = np.load(mip_path)  # [H, W], float32, 范围[0,1]

        # 确保尺寸正确
        if mip_array.shape != (self.image_size, self.image_size):
            # 使用scipy进行resize（保持精度）
            from scipy.ndimage import zoom
            zoom_factors = (self.image_size / mip_array.shape[0],
                            self.image_size / mip_array.shape[1])
            mip_array = zoom(mip_array, zoom_factors, order=1)

        # 转换为torch张量并添加通道维度
        mip_tensor = torch.from_numpy(mip_array).float().unsqueeze(0)  # [1, H, W]

        return mip_tensor

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        item = self.image_list[idx]

        # 加载造影图像（目标）
        angio_image = Image.open(item['image_path']).convert('L')
        target_tensor = self.transform(angio_image)  # [1, H, W]

        # 加载预计算的MIP掩码（.npy格式，高精度）
        mask_mip = self._load_mip_mask(item['mip_path'])  # [1, H, W]

        # 解析角度信息（从文件名）
        angle_info = get_angle_info(item['img_name'])

        return {
            'target': target_tensor,                      # 目标造影图 [1, H, W]
            'mask': mask_mip,                             # 条件：2D MIP掩码 [1, H, W] (float32)
            'angle_matrix': angle_info['rotation_matrix'], # 旋转矩阵 [9]
            'angle_deg': angle_info['angle_deg'],         # 角度（度数）
            'view_index': angle_info['view_index'],       # 视角序号
            'patient': item['patient'],
            'img_name': item['img_name']
        }


def create_dataloader(shuffle=True):
    """
    创建DataLoader
    """
    config = Config()

    dataset = PulmonaryAngiographyDataset(
        root_dir=config.data_dir,
        image_size=config.image_size,
        device='cpu'
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    return dataloader


# 测试代码
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    config = Config()

    # 使用脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..")  # 假设dataset.py在utils/下

    if os.path.exists(os.path.join(data_dir, "angiographs")):
        dataset = PulmonaryAngiographyDataset(
            root_dir=data_dir,
            image_size=config.image_size
        )

        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Target形状: {sample['target'].shape}")
            print(f"Mask形状: {sample['mask'].shape}")
            print(f"Mask数据类型: {sample['mask'].dtype}")
            print(f"Mask范围: [{sample['mask'].min():.6f}, {sample['mask'].max():.6f}]")
            print(f"Mask精度: {sample['mask'][0, 100, 100].item():.8f}")  # 显示高精度值

            # 可视化
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            target_disp = (sample['target'][0].numpy() + 1) / 2
            mask_disp = sample['mask'][0].numpy()

            axes[0].imshow(target_disp, cmap='gray')
            axes[0].set_title("造影图像 (目标)")
            axes[0].axis('off')

            im = axes[1].imshow(mask_disp, cmap='hot')
            axes[1].set_title("MIP掩码 (条件)\n.npy格式, float32精度")
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046)

            plt.tight_layout()
            plt.show()
        else:
            print("没有找到数据，请先运行 preprocess_masks.py")
    else:
        print(f"目录不存在，请确认路径")