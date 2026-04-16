"""
预处理脚本：将3D血管掩码批量转换为2D MIP并保存为.npy文件
使用方法：python preprocess_masks.py
"""

import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from tqdm import tqdm
from config import Config


def load_mask_and_mip_to_array(nifti_path, target_size=(256, 256), projection_axis=0):
    """
    加载3D NIfTI掩码并转换为2D MIP，返回numpy数组

    Args:
        nifti_path: .nii或.nii.gz文件路径
        target_size: 输出2D图像尺寸 (H, W)
        projection_axis: 投影轴 (0=X轴/前后方向)

    Returns:
        mip_2d: [H, W] 归一化到[0,1]的numpy数组（float32）
    """
    # 加载NIfTI
    nii = nib.load(nifti_path)
    mask_3d = nii.get_fdata()

    # MIP投影
    mip_2d = np.max(mask_3d, axis=projection_axis)

    # 归一化到[0,1]
    if mip_2d.max() > 0:
        mip_2d = mip_2d / mip_2d.max()

    # 调整尺寸到目标大小
    if mip_2d.shape != target_size:
        zoom_factors = (target_size[0] / mip_2d.shape[0],
                        target_size[1] / mip_2d.shape[1])
        mip_2d = zoom(mip_2d, zoom_factors, order=1)

    # 确保值在[0,1]范围内
    mip_2d = np.clip(mip_2d, 0, 1)

    # 逆时针旋转90度
    mip_2d = np.rot90(mip_2d, k=1)  # k=1 表示逆时针旋转90度

    return mip_2d.astype(np.float32)


def preprocess_all_masks(config, overwrite=False):
    """
    预处理所有患者的3D掩码

    Args:
        config: 配置对象
        overwrite: 是否覆盖已存在的文件
    """
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 基于脚本位置构建完整路径
    masks_dir = os.path.join(script_dir, "mask")      # 原始3D掩码目录
    output_dir = os.path.join(script_dir, "mask2D")   # 输出2D MIP目录（.npy格式）

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 检查原始目录是否存在
    if not os.path.exists(masks_dir):
        print(f"错误: 原始掩码目录不存在: {masks_dir}")
        print(f"当前脚本位置: {script_dir}")
        print(f"请确认目录结构是否正确")
        return

    # 收集所有.nii/.nii.gz文件
    mask_files = []
    for ext in ['.nii.gz', '.nii']:
        for f in os.listdir(masks_dir):
            if f.endswith(ext):
                mask_files.append(os.path.join(masks_dir, f))

    if len(mask_files) == 0:
        print(f"错误: 在 {masks_dir} 中没有找到掩码文件")
        return

    print(f"找到 {len(mask_files)} 个3D掩码文件")
    print(f"输出目录: {output_dir}")
    print(f"输出格式: .npy (float32, 保留原始精度)")
    print(f"目标尺寸: {config.image_size}x{config.image_size}")
    print(f"投影轴: {config.mip_projection_axis} (0=X轴/前后)")
    print(f"旋转: 逆时针90度")
    print("=" * 50)

    # 统计
    converted = 0
    skipped = 0

    # 批量处理
    for mask_path in tqdm(mask_files, desc="转换MIP"):
        # 生成输出文件名（使用.npy扩展名）
        base_name = os.path.basename(mask_path)
        # 移除扩展名
        for ext in ['.nii.gz', '.nii']:
            if base_name.endswith(ext):
                base_name = base_name[:-len(ext)]
                break
        output_path = os.path.join(output_dir, f"{base_name}_mip.npy")

        # 检查是否已存在
        if os.path.exists(output_path) and not overwrite:
            skipped += 1
            continue

        try:
            # 转换MIP
            mip_array = load_mask_and_mip_to_array(
                mask_path,
                target_size=(config.image_size, config.image_size),
                projection_axis=config.mip_projection_axis
            )

            # 保存为.npy文件（保留float32精度）
            np.save(output_path, mip_array)
            converted += 1

        except Exception as e:
            print(f"  错误: 处理 {mask_path} 失败: {e}")

    # 打印统计
    print("=" * 50)
    print(f"预处理完成!")
    print(f"  转换: {converted} 个")
    print(f"  跳过: {skipped} 个")
    print(f"  输出目录: {output_dir}")
    print("=" * 50)


def verify_mip_output(config):
    """
    验证MIP输出结果
    """
    import matplotlib.pyplot as plt

    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    masks_dir = os.path.join(script_dir, "mask")
    mip_dir = os.path.join(script_dir, "mask2D")

    if not os.path.exists(mip_dir):
        print("请先运行 preprocess_masks.py")
        return

    # 查找.npy文件
    mip_files = [f for f in os.listdir(mip_dir) if f.endswith('_mip.npy')]
    if len(mip_files) == 0:
        print("没有找到MIP输出文件（.npy）")
        return

    # 显示前3个示例
    num_examples = min(3, len(mip_files))
    fig, axes = plt.subplots(2, num_examples, figsize=(4*num_examples, 8))

    for i in range(num_examples):
        mip_file = mip_files[i]
        patient_name = mip_file.replace('_mip.npy', '')

        # 加载MIP结果（.npy格式）
        mip_path = os.path.join(mip_dir, mip_file)
        mip_array = np.load(mip_path)  # 直接加载，保持float32精度

        # 显示时转换为0-255范围
        mip_display = mip_array  # 已经是0-1范围

        # 尝试加载原始3D掩码的中间切片（仅用于对比）
        mask_path = None
        for ext in ['.nii.gz', '.nii']:
            candidate = os.path.join(masks_dir, f"{patient_name}{ext}")
            if os.path.exists(candidate):
                mask_path = candidate
                break

        if mask_path:
            nii = nib.load(mask_path)
            mask_3d = nii.get_fdata()
            # 取中间切片
            mid_slice = mask_3d.shape[0] // 2
            slice_2d = mask_3d[mid_slice, :, :]
            if slice_2d.max() > 0:
                slice_2d = slice_2d / slice_2d.max()

        # 显示
        axes[0, i].imshow(slice_2d if mask_path else np.zeros_like(mip_display), cmap='gray')
        axes[0, i].set_title(f"原始3D掩码 (中间切片)\n{patient_name}")
        axes[0, i].axis('off')

        axes[1, i].imshow(mip_display, cmap='hot')
        axes[1, i].set_title(f"MIP投影 (2D) .npy\n{patient_name}")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig("./mip_verification.png", dpi=150)
    plt.show()
    print("验证图已保存: ./mip_verification.png")


if __name__ == "__main__":
    config = Config()

    print("=" * 60)
    print("3D掩码 → 2D MIP 预处理工具 (.npy格式)")
    print("=" * 60)
    print(f"原始掩码: ./mask/")
    print(f"输出目录: ./mask2D/")
    print(f"输出格式: .npy (float32, 无损精度)")
    print("=" * 60)

    # 执行预处理
    preprocess_all_masks(config, overwrite=False)

    # 可选：验证结果
    print("\n是否查看验证结果？(y/n)")
    choice = input().strip().lower()
    if choice == 'y':
        verify_mip_output(config)