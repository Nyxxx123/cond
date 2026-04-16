"""
角度处理工具：从文件名解析角度，生成旋转矩阵
"""
import re
import torch
import math


def parse_angle_from_filename(filename):
    """
    从文件名中解析视角序号并计算旋转角度

    文件名格式示例: "PARSE001_3_mask.png" 或 "img_005_2.png"
    规则: 下划线分隔，倒数第二个或某个位置是视角序号

    Args:
        filename: 图像文件名

    Returns:
        angle_deg: 旋转角度（度数）
        view_index: 视角序号
    """
    # 尝试多种模式匹配视角序号
    # 模式1: _数字_ (如 _3_)
    pattern1 = r'_(\d+)_'
    match = re.search(pattern1, filename)

    if match:
        view_index = int(match.group(1))
    else:
        # 模式2: 文件名末尾数字 (如 _3.png)
        pattern2 = r'_(\d+)\.'
        match = re.search(pattern2, filename)
        if match:
            view_index = int(match.group(1))
        else:
            # 默认值，返回0度
            print(f"\nWarning: Cannot parse angle from filename '{filename}', using default 0")
            view_index = 0

    # 计算角度: 视角序号 × 22.5°
    angle_deg = view_index * 22.5
    angle_rad = math.radians(angle_deg)

    return angle_deg, angle_rad, view_index


def get_rotation_matrix_z(angle_rad):
    """
    生成绕Z轴的旋转矩阵（3x3）

    Args:
        angle_rad: 旋转角度（弧度）

    Returns:
        R: 3x3 torch张量
    """
    cos = math.cos(angle_rad)
    sin = math.sin(angle_rad)

    R = torch.tensor([
        [cos, -sin, 0],
        [sin, cos, 0],
        [0, 0, 1]
    ], dtype=torch.float32)

    return R


def get_rotation_matrix_9d(angle_rad):
    """
    生成绕Z轴的旋转矩阵并展平为9维向量

    Args:
        angle_rad: 旋转角度（弧度）

    Returns:
        R_9d: 9维torch张量
    """
    R = get_rotation_matrix_z(angle_rad)
    return R.flatten()


def get_angle_info(filename):
    """
    从文件名获取完整的角度信息

    Args:
        filename: 图像文件名

    Returns:
        dict: 包含 angle_deg, angle_rad, view_index, rotation_matrix_9d
    """
    angle_deg, angle_rad, view_index = parse_angle_from_filename(filename)
    rotation_matrix_9d = get_rotation_matrix_9d(angle_rad)

    return {
        'angle_deg': angle_deg,
        'angle_rad': angle_rad,
        'view_index': view_index,
        'rotation_matrix': rotation_matrix_9d
    }


def test_angle_utils():
    """测试角度工具函数"""
    test_files = [
        "PARSE001_0_mask.png",
        "PARSE001_1_mask.png",
        "PARSE001_2_mask.png",
        "PARSE001_3_mask.png",
        "PARSE001_4_mask.png",
        "img_005_5.png",
        "unknown_name.png"
    ]

    print("=" * 50)
    print("Testing angle utilities")
    print("=" * 50)

    for fileName in test_files:
        info = get_angle_info(fileName)
        print(f"\nFile: {fileName}")
        print(f"  View index: {info['view_index']}")
        print(f"  Angle: {info['angle_deg']}° ({info['angle_rad']:.4f} rad)")
        print(f"  Rotation matrix (9D): {info['rotation_matrix'].tolist()}")


if __name__ == "__main__":
    test_angle_utils()