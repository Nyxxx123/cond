"""
配置文件：包含所有超参数
"""

import torch


class Config:
    # 设备设置
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 数据参数 - CT图像
    batch_size = 4
    image_size = 256  # CT图像尺寸
    channels = 1  # 灰度图
    data_dir = "./PARSE"  # CT图像目录（支持子文件夹结构）
    mip_cache_dir = "mask2D"  # MIP缓存目录名（相对于data_dir）

    # 采样参数（新增 DDIM 配置）
    sampler_type = "ddpm"      # "ddpm" 或 "ddim"
    ddim_steps = 50            # DDIM 采样步数（20-100，越高质量越好但越慢）
    ddim_eta = 0.0             # DDIM 随机性（0=确定性，1=DDPM风格）

    #掩码适配参数
    mask_type = "3d"  # "2d" 或 "3d"：使用2D MIP还是3D原始掩码
    mask_3d_size = (64, 64, 64)  # 3D掩码下采样尺寸 (D, H, W)
    # MIP投影参数
    mip_projection_axis = 2  # 0=X轴（前后方向），1=Y轴，2=Z轴 （右手系）

    # 条件参数
    cond_dim = 256  # 条件向量维度
    cond_block_type = "cross_attention"  # "add" 或 "cross_attention"

    # 扩散参数
    timesteps = 1000
    beta_start = 0.0001
    beta_end = 0.02
    schedule_type = "linear"  # "linear" 或 "cosine"

    # 模型参数
    base_channels = 32
    time_emb_dim = 256
    dropout = 0.1

    # 训练参数
    num_epochs = 1000
    learning_rate = 2e-4
    weight_decay = 1e-4
    grad_clip = 1.0

    # 采样参数
    sample_batch_size = 16
    sample_frequency = 25  # 每5个epoch采样一次

    # 日志和保存
    checkpoint_dir = "./checkpoints_ct_3d"
    sample_dir = "./samples_ct_3d"