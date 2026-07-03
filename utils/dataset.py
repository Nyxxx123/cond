"""
条件数据集：肺动脉造影 + 血管掩码（支持2D MIP或3D原始掩码）
数据组织方式：
    2D模式 (mask_type="2d"):
        data/
        ├── angiographs/                    # 肺动脉造影图像（目标）
        │   ├── patient001/
        │   │   ├── patient001_0_mask.png
        │   │   └── ...
        │   └── patient002/
        ├── non_angiographs/                # 无造影CT图像（可选条件）
        │   ├── patient001/
        │   │   ├── patient001_0_mask.png
        │   │   └── ...
        │   └── patient002/
        └── mask2D/                         # 预计算的2D MIP掩码（.npy格式）
            ├── patient001_mip.npy
            └── patient002_mip.npy

    3D模式 (mask_type="3d"):
        data/
        ├── angiographs/                    # 肺动脉造影图像（目标）
        │   ├── patient001/
        │   │   ├── patient001_0_mask.png
        │   │   └── ...
        │   └── patient002/
        ├── non_angiographs/                # 无造影CT图像（可选条件）
        │   ├── patient001/
        │   │   ├── patient001_0_mask.png
        │   │   └── ...
        │   └── patient002/
        └── mask/                           # 原始3D血管掩码（.nii.gz）
            ├── patient001.nii.gz
            └── patient002.nii.gz
"""

import os
import re
import numpy as np
import nibabel as nib
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from config import Config
from utils.analyse_angle import get_angle_info


class PulmonaryAngiographyDataset(Dataset):
    """
    肺动脉造影数据集
    """

    def __init__(self, root_dir, image_size=256, mask_type="3d",
                 mask_3d_size=(64, 64, 64), transform=None, device='cpu',
                 use_non_angio=True):
        """
        Args:
            root_dir: 数据根目录
            image_size: 输出2D图像尺寸
            mask_type: "2d" 或 "3d"
            mask_3d_size: 3D掩码下采样尺寸 (D, H, W)（仅3D模式使用）
            transform: 图像变换
            device: 设备
            use_non_angio: 是否使用无造影CT作为条件
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.mask_type = mask_type
        self.mask_3d_size = mask_3d_size
        self.device = device
        self.use_non_angio = use_non_angio

        # 目录路径
        self.angiographs_dir = os.path.join(root_dir, 'angiographs')
        if use_non_angio:
            self.non_angiographs_dir = os.path.join(root_dir, 'non_angiographs')

        # 根据类型选择掩码目录
        if mask_type == "2d":
            self.mask_dir = os.path.join(root_dir, 'mask2D')
        else:
            self.mask_dir = os.path.join(root_dir, 'mask')

        # 存储所有图像路径
        self.image_list = []

        # 收集数据
        self._collect_data()

        print(f"找到 {len(self.image_list)} 张造影图像")
        print(f"掩码类型: {mask_type}")
        print(f"掩码目录: {self.mask_dir}")
        if use_non_angio:
            print(f"无造影CT目录: {self.non_angiographs_dir}")

        # 定义图像变换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
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
        """收集所有造影图像和对应的掩码路径（以及无造影CT路径）"""
        if not os.path.exists(self.angiographs_dir):
            raise FileNotFoundError(f"目录不存在: {self.angiographs_dir}")

        if not os.path.exists(self.mask_dir):
            raise FileNotFoundError(f"掩码目录不存在: {self.mask_dir}")

        if self.use_non_angio and not os.path.exists(self.non_angiographs_dir):
            raise FileNotFoundError(f"无造影CT目录不存在: {self.non_angiographs_dir}")

        # 遍历患者文件夹
        patient_folders = sorted(os.listdir(self.angiographs_dir))

        for patient_folder in patient_folders:
            patient_angio_dir = os.path.join(self.angiographs_dir, patient_folder)

            if not os.path.isdir(patient_angio_dir):
                continue

            # 根据类型查找对应的掩码文件
            if self.mask_type == "2d":
                mask_path = os.path.join(self.mask_dir, f"{patient_folder}_mip.npy")
                if not os.path.exists(mask_path):
                    print(f"警告: 患者 {patient_folder} 的MIP文件不存在，跳过")
                    continue
            else:
                mask_path = None
                for ext in ['.nii.gz', '.nii']:
                    candidate = os.path.join(self.mask_dir, f"{patient_folder}{ext}")
                    if os.path.exists(candidate):
                        mask_path = candidate
                        break
                if mask_path is None:
                    print(f"警告: 患者 {patient_folder} 的3D掩码不存在，跳过")
                    continue

            # 收集该患者的所有造影图像
            image_files = sorted(os.listdir(patient_angio_dir))
            supported_ext = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}

            for img_file in image_files:
                ext = os.path.splitext(img_file)[1].lower()
                if ext not in supported_ext:
                    continue

                try:
                    patient_id, view_index = self._extract_info_from_filename(img_file)

                    if patient_id != patient_folder:
                        print(f"警告: 文件名中的患者ID({patient_id})与文件夹名({patient_folder})不一致，跳过")
                        continue
                except ValueError as e:
                    print(f"警告: 跳过文件 {img_file}，{e}")
                    continue

                # 检查无造影CT文件是否存在
                non_angio_path = None
                if self.use_non_angio:
                    non_angio_path = os.path.join(self.non_angiographs_dir, patient_folder, img_file)
                    if not os.path.exists(non_angio_path):
                        print(f"警告: 患者 {patient_folder} 的无造影CT文件 {img_file} 不存在，跳过该样本")
                        continue

                self.image_list.append({
                    'image_path': os.path.join(patient_angio_dir, img_file),
                    'mask_path': mask_path,
                    'non_angio_path': non_angio_path,
                    'patient': patient_folder,
                    'img_name': img_file,
                    'view_index': view_index
                })

    def _load_mask(self, mask_path):
        """
        加载掩码（根据类型选择加载方式）
        """
        if self.mask_type == "2d":
            # 2D模式：加载.npy文件
            mip_array = np.load(mask_path)  # [H, W], float32, 范围[0,1]

            # 确保尺寸正确
            if mip_array.shape != (self.image_size, self.image_size):
                from scipy.ndimage import zoom
                zoom_factors = (self.image_size / mip_array.shape[0],
                                self.image_size / mip_array.shape[1])
                mip_array = zoom(mip_array, zoom_factors, order=1)

            # 转换为torch张量并添加通道维度
            mask_tensor = torch.from_numpy(mip_array).float().unsqueeze(0)  # [1, H, W]
            return mask_tensor

        else:
            # 3D模式：加载.nii.gz文件
            nii = nib.load(mask_path)
            mask_3d = nii.get_fdata()
            mask_3d = mask_3d[np.newaxis, ...]  # [1, D, H, W]
            mask_tensor = torch.from_numpy(mask_3d).float()

            # 下采样到目标尺寸
            if mask_tensor.shape[1:] != self.mask_3d_size:
                mask_tensor = mask_tensor.unsqueeze(0)
                mask_tensor = F.interpolate(
                    mask_tensor,
                    size=self.mask_3d_size,
                    mode='trilinear',
                    align_corners=False
                )
                mask_tensor = mask_tensor.squeeze(0)

            # 归一化到[0,1]
            if mask_tensor.max() > 0:
                mask_tensor = mask_tensor / mask_tensor.max()

            return mask_tensor

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        item = self.image_list[idx]

        # 加载造影图像（目标）
        angio_image = Image.open(item['image_path']).convert('L')
        target_tensor = self.transform(angio_image)

        # 加载掩码
        mask = self._load_mask(item['mask_path'])

        # 加载无造影CT（条件）
        if self.use_non_angio and item['non_angio_path'] is not None:
            non_angio_image = Image.open(item['non_angio_path']).convert('L')
            non_angio_tensor = self.transform(non_angio_image)
        else:
            # 占位符（全零），与 target 同形状
            non_angio_tensor = torch.zeros_like(target_tensor)

        # 解析角度信息（使用四元数表示）
        from config import Config
        config = Config()
        angle_info = get_angle_info(item['img_name'], angle_rep=config.angle_rep)

        return {
            'target': target_tensor,
            'mask': mask,
            'non_angio': non_angio_tensor,
            'angle': torch.from_numpy(angle_info['angle_vector']).float(),  # 四元数 [4]
            'angle_deg': angle_info['angle_deg'],
            'view_index': angle_info['view_index'],
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
        mask_type=config.mask_type,
        mask_3d_size=config.mask_3d_size,
        device='cpu',
        use_non_angio=config.use_non_angio
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
