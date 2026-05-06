"""
配置文件：包含所有超参数
"""

import torch


class Config:
    # 设备设置
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 预测类型: "epsilon" 或 "v"
    prediction_type = "v"  # 推荐使用 "v"，更稳定

    # EMA配置
    ema_decay = 0.9999  # 衰减率，推荐0.999-0.9999
    use_ema = False  # 是否使用EMA

    # LPIPS感知损失配置
    use_lpips = True  # 开关：True=启用LPIPS
    lpips_loss_weight = 0.1  # LPIPS损失权重（建议0.05-0.5）
    lpips_net = "alex"  # 可选: "alex", "vgg"

    # GAN配置
    use_gan = False  # 开关：True=启用GAN
    gan_loss_weight = 0.05  # GAN损失权重
    disc_lr = 5e-5  # 判别器学习率
    gan_start_epoch = 100

    # 数据参数 - CT图像
    batch_size = 4
    image_size = 256  # CT图像尺寸
    channels = 1  # 灰度图
    data_dir = "./PARSE"  # CT图像目录（支持子文件夹结构）
    mip_cache_dir = "mask2D"  # MIP缓存目录名（相对于data_dir）

    # ========== 新增：无造影CT条件配置 ==========
    use_non_angio = True          # 是否使用无造影CT图作为条件
    # =======================================

    # 采样参数（新增 DDIM 配置）
    sampler_type = "ddpm"      # "ddpm" 或 "ddim"
    ddim_steps = 50            # DDIM 采样步数（20-100，越高质量越好但越慢）
    ddim_eta = 0.0             # DDIM 随机性（0=确定性，1=DDPM风格）

    #掩码适配参数
    mask_type = "3d"  # "2d" 或 "3d"：使用2D MIP还是3D原始掩码
    mask_3d_size = (64, 64, 64)  # 3D掩码下采样尺寸 (D, H, W)
    # MIP投影参数
    mip_projection_axis = 2  # 0=X轴（前后方向），1=Y轴，2=Z轴 （右手系）

    # 角度条件配置
    use_angle_condition = True  # 是否使用角度条件
    angle_rep = "quaternion"  # 角度表示方式: "quaternion"(四元数) 或 "rotation_matrix"(9维) 或 "euler"(欧拉角)
    angle_dim = 4  # 四元数维度

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
    num_epochs = 1500
    learning_rate = 2e-4
    weight_decay = 1e-4
    grad_clip = 1.0

    # 采样参数
    sample_batch_size = 16
    sample_frequency = 25  # 每5个epoch采样一次

    # 日志和保存
    checkpoint_dir = "./checkpoints_ct_all_infor-v"
    sample_dir = "./samples_ct_all_infor-v"