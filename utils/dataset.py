import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class CTDataset(Dataset):
    """
    CT图像数据集
    文件夹结构：
        PARSE/
        ├── patient1/           # 患者1文件夹（可选，支持子文件夹）
        │   ├── ct_001.png
        │   ├── ct_002.png
        │   └── ...
        ├── patient2/           # 患者2文件夹
        │   ├── ct_001.png
        │   └── ...
        └── ct_image.png        # 也支持直接放在根目录
    """
    def __init__(self, root_dir, image_size=256, transform=None):
        """
        Args:
            root_dir: PARSE目录路径
            image_size: 目标图像尺寸
            transform: 自定义转换（如果不提供，使用默认）
        """
        self.root_dir = root_dir
        self.image_size = image_size

        # 存储所有图像路径
        self.image_paths = []

        # 递归收集所有图像
        self._collect_images(root_dir)

        print(f"找到 {len(self.image_paths)} 张CT图像")

        # 定义默认转换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])  # 归一化到[-1,1]
            ])
        else:
            self.transform = transform

    def _collect_images(self, directory):
        """
        递归收集目录下的所有图像
        """
        # 支持的图像格式
        supported_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

        for root, dirs, files in os.walk(directory):
            for file in files:
                # 检查文件扩展名
                ext = os.path.splitext(file)[1].lower()
                if ext in supported_extensions:
                    full_path = os.path.join(root, file)
                    self.image_paths.append(full_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        返回：
            image: CT图像 [1, H, W] 归一化到[-1,1]
            dummy_label: 0（保持接口兼容）
        """
        img_path = self.image_paths[idx]

        # 加载图像（转为灰度图）
        image = Image.open(img_path).convert('L')

        # 应用转换
        image = self.transform(image)

        # 返回图像和虚拟标签（保持与原有训练代码接口一致）
        return image, 0

    def get_stats(self, num_samples=100):
        """
        可选：计算数据集的统计信息
        """
        print("计算数据集统计信息...")
        all_images = []
        for i in range(min(num_samples, len(self))):
            img, _ = self[i]
            all_images.append(img.numpy())

        import numpy as np
        all_images = np.array(all_images)
        print(f"  Mean: {all_images.mean():.4f}")
        print(f"  Std: {all_images.std():.4f}")
        print(f"  Min: {all_images.min():.4f}")
        print(f"  Max: {all_images.max():.4f}")
        return all_images.mean(), all_images.std()


def create_ct_dataloader(config, shuffle=True):
    """
    创建CT数据集的DataLoader
    """
    dataset = CTDataset(
        root_dir=config.data_dir,
        image_size=config.image_size
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        drop_last=True  # 丢弃最后一个不完整的batch
    )

    return dataloader


# 测试代码
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=" * 50)
    print("测试CT数据集...")
    print("=" * 50)

    # 测试配置
    class TestConfig:
        data_dir = "../PARSE"
        image_size = 512
        batch_size = 4

    config = TestConfig()

    # 检查目录
    if not os.path.exists(config.data_dir):
        print(f"错误: 目录 {config.data_dir} 不存在！")
        print("请创建 PARSE 目录并放入CT图像")
        exit(1)

    # 创建数据集
    dataset = CTDataset(config.data_dir, config.image_size)

    if len(dataset) == 0:
        print(f"错误: 在 {config.data_dir} 中没有找到图像！")
        print("请确保目录下包含 PNG/JPG 等格式的CT图像")
        exit(1)

    # 创建DataLoader
    dataloader = create_ct_dataloader(config)

    # 获取一个batch
    for images, labels in dataloader:
        print(f"Batch形状: {images.shape}")
        print(f"标签形状: {labels.shape}")
        print(f"数值范围: [{images.min():.3f}, {images.max():.3f}]")

        # 可视化
        fig, axes = plt.subplots(1, min(4, images.shape[0]), figsize=(12, 3))
        if images.shape[0] == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            # 反归一化到[0,1]显示
            img_display = (images[i, 0].numpy() + 1) / 2
            ax.imshow(img_display, cmap='gray')
            ax.axis('off')
            ax.set_title(f"CT {i+1}")

        plt.tight_layout()
        plt.savefig("./ct_sample.png", dpi=150)
        print(f"\n已保存示例到 ./ct_sample.png")
        plt.show()
        break