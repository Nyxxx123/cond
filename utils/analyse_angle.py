"""
角度处理工具：从文件名解析角度，生成四元数/旋转矩阵
"""
import re
import torch
import math
import numpy as np


def parse_angle_from_filename(filename):
    """
    从文件名中解析视角序号并计算旋转角度
    格式: "patient001_0_mask.png" -> view_index=0 (0°)
          "patient001_1_mask.png" -> view_index=1 (22.5°)
    """
    pattern = r'_(\d+)_'
    match = re.search(pattern, filename)

    if match:
        view_index = int(match.group(1))
    else:
        pattern = r'_(\d+)\.'
        match = re.search(pattern, filename)
        if match:
            view_index = int(match.group(1))
        else:
            print(f"Warning: Cannot parse angle from '{filename}', using 0")
            view_index = 0

    angle_deg = view_index * 22.5
    angle_rad = math.radians(angle_deg)

    return angle_deg, angle_rad, view_index


def euler_to_quaternion(angle_rad):
    """
    欧拉角转四元数（绕Z轴旋转）
    Args:
        angle_rad: 绕Z轴的旋转角度（弧度）
    Returns:
        quaternion: [w, x, y, z] 四元数，形状为 (4,)
    """
    # 绕Z轴旋转的四元数
    half_angle = angle_rad / 2
    w = math.cos(half_angle)
    x = 0.0
    y = 0.0
    z = math.sin(half_angle)

    return np.array([w, x, y, z], dtype=np.float32)


def rotation_matrix_to_quaternion(R):
    """
    旋转矩阵转四元数
    Args:
        R: 3x3 旋转矩阵
    Returns:
        quaternion: [w, x, y, z] 四元数
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    return np.array([w, x, y, z], dtype=np.float32)


def get_rotation_matrix_z(angle_rad):
    """生成绕Z轴的旋转矩阵（3x3）"""
    cos = math.cos(angle_rad)
    sin = math.sin(angle_rad)
    R = torch.tensor([
        [cos, -sin, 0],
        [sin, cos, 0],
        [0, 0, 1]
    ], dtype=torch.float32)
    return R


def get_angle_info(filename, angle_rep="quaternion"):
    """
    从文件名获取完整的角度信息

    Args:
        filename: 图像文件名
        angle_rep: 角度表示方式 - "quaternion"(四元数), "rotation_matrix"(9维), "euler"(欧拉角)

    Returns:
        dict: 包含 angle_deg, angle_rad, view_index, angle_vector
    """
    angle_deg, angle_rad, view_index = parse_angle_from_filename(filename)

    if angle_rep == "quaternion":
        # 四元数 [w, x, y, z]
        angle_vector = euler_to_quaternion(angle_rad)
    elif angle_rep == "rotation_matrix":
        # 9维旋转矩阵
        R = get_rotation_matrix_z(angle_rad)
        angle_vector = R.flatten().numpy()
    elif angle_rep == "euler":
        # 欧拉角 [roll, pitch, yaw]，这里只有yaw
        angle_vector = np.array([0.0, 0.0, angle_rad], dtype=np.float32)
    else:
        raise ValueError(f"Unknown angle_rep: {angle_rep}")

    return {
        'angle_deg': angle_deg,
        'angle_rad': angle_rad,
        'view_index': view_index,
        'angle_vector': angle_vector,  # 统一名称
        'rotation_matrix': get_rotation_matrix_z(angle_rad).numpy()  # 保留兼容
    }


# 测试代码
if __name__ == "__main__":
    test_files = ["patient001_0_mask.png", "patient001_1_mask.png", "patient001_2_mask.png"]

    for f in test_files:
        info = get_angle_info(f, angle_rep="quaternion")
        print(f"{f}: view={info['view_index']}, angle={info['angle_deg']}°, quaternion={info['angle_vector']}")