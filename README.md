# Conditional Diffusion Model for Pulmonary Angiography Synthesis

基于条件扩散模型的**肺动脉造影 X 光图像生成**。以血管掩码（3D/2D MIP）、投影角度（四元数）以及无造影 CT 作为条件，生成逼真的造影增强 X 光投影图像。

## 项目结构

```
├── config.py                     # 所有超参数配置
├── models/
│   ├── cond_unet.py              # 条件 U-Net（4 层，含 cross-attention 条件注入）
│   ├── encoder.py                # 多条件编码器（掩码 + 角度 + 无造影 CT）
│   ├── discriminator.py          # PatchGAN 判别器 / 条件判别器
│   ├── cond_attention.py         # 交叉注意力模块
│   └── unet.py                   # 基础 U-Net
├── utils/
│   ├── diffusion.py              # 扩散过程（DDPM/DDIM 采样、v/ε 预测）
│   ├── noise_schedule.py         # 噪声调度（linear / cosine）
│   ├── dataset.py                # 数据加载器
│   ├── ema.py                    # EMA 模型
│   └── analyse_angle.py          # 角度解析工具
├── train_final.py                # 训练（MSE + LPIPS + MS-SSIM + GAN）
├── sample_conditional.py         # 推理采样
├── evaluate.py                   # 评估（PSNR/SSIM/FID/KID/LPIPS）
└── samples/                      # 训练过程中的采样结果
```

## 方法概述

### 数据流

```
CT 体数据 → DeepDRR 渲染器 → 多角度 2D 投影（训练数据）
训练数据 → 条件 U-Net 扩散模型 → 生成造影图像
```

### 条件输入

模型支持多种条件组合，通过 `config.py` 灵活配置：

| 条件类型 | 编码器 | 说明 |
|---------|--------|------|
| **血管掩码** | `MaskEncoder2D` / `MaskEncoder3D` | 2D MIP 投影或 3D 体素掩码 |
| **投影角度** | `AngleEncoder` | 四元数/旋转矩阵/欧拉角表示 |
| **无造影 CT** | `XRayEncoder` | 平扫 CT 的 DRR 投影图像 |

### 模型架构

- **ConditionalUNet**：4 层 U-Net，通过 cross-attention 或加法方式注入时间嵌入和条件特征
- **MultiConditionEncoder**：融合掩码、角度、无造影 CT 三种条件，输出统一的条件向量
- **CondDiscriminator**：条件 PatchGAN，以融合条件向量作为判别条件

### 损失函数

| 损失 | 权重 | 说明 |
|------|------|------|
| MSE | 1.0（固定） | 扩散噪声/速度预测的基准损失 |
| LPIPS | 0.5 | 感知损失（AlexNet 主干） |
| MS-SSIM | 0.3 | 多尺度结构相似性损失 |
| GAN | 0.1 | 条件 PatchGAN 对抗损失 |

### 采样方法

- **DDPM**：标准去噪扩散概率模型采样（1000 步）
- **DDIM**：确定性加速采样（可配置步数，如 50 步）

## 快速开始

### 环境要求

- **CUDA GPU**（DeepDRR 依赖 PyCUDA）
- Python ≥ 3.7

```bash
# DeepDRR 依赖
pip install pycuda numpy torch torchvision nibabel pydicom scikit-image scipy

# 扩散模型依赖
pip install lpips torchmetrics tqdm matplotlib
```

### 数据准备

数据目录结构（由 `config.py` 中的 `data_dir` 配置，默认为 `./PARSE`）：

```
PARSE/
├── angiographs/                  # 目标造影图像
│   └── patient001/
│       ├── patient001_0_mask.png
│       └── patient001_1_mask.png
├── non_angiographs/              # 无造影 CT 投影（可选）
│   └── patient001/
│       ├── patient001_0_mask.png
│       └── patient001_1_mask.png
├── mask/                         # 3D 血管掩码 (.nii.gz)
│   └── patient001.nii.gz
└── mask2D/                       # 预计算 2D MIP 掩码 (.npy，可选)
    └── patient001_mip.npy
```

测试数据目录 `TEST/` 结构相同，用于推理评估。

### 训练

所有超参数在 `config.py` 的 `Config` 类中配置，修改后直接运行训练脚本：

```bash
# 完整模型
python train_final.py
```

模型保存为字典（含 `epoch`, `model_state_dict`, `optimizer_state_dict`, `loss`, `config_dict`）。

### 推理

```bash
python sample_conditional.py
```

推理流程：
1. 扫描 `TEST/` 文件夹获取掩码 + 造影 + 无造影文件
2. 加载训练好的模型
3. 对每个患者-视角对生成多张样本
4. 保存单张样本、网格图、GT vs 生成对比图到 `Generated-*/` 目录

### 评估

```bash
python evaluate.py
```

自动配对 `TEST/angiographs/` 中的真实图像与 `Generated*/` 中的生成图像，计算以下指标：
- **PSNR** — 峰值信噪比
- **SSIM** — 结构相似性
- **MS-SSIM** — 多尺度结构相似性
- **LPIPS** — 学习感知图像块相似度
- **FID** — Fréchet Inception Distance
- **KID** — Kernel Inception Distance

## 关键配置

在 `config.py` 中修改以下配置项：

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `prediction_type` | `"v"` | `"epsilon"` (DDPM) 或 `"v"` (v-prediction) |
| `sampler_type` | `"ddpm"` | `"ddpm"` 或 `"ddim"` |
| `ddim_steps` | `50` | DDIM 采样步数 |
| `mask_type` | `"3d"` | `"2d"` (MIP) 或 `"3d"` (体素掩码) |
| `use_angle_condition` | `True` | 是否使用角度条件 |
| `angle_rep` | `"quaternion"` | 角度表示：`"quaternion"` / `"rotation_matrix"` / `"euler"` |
| `use_non_angio` | `True` | 是否使用无造影 CT 条件 |
| `use_lpips` | `True` | 是否启用 LPIPS 损失 |
| `use_ms_ssim` | `True` | 是否启用 MS-SSIM 损失 |
| `use_gan` | `False` | 是否启用 GAN 损失 |
| `gan_start_epoch` | `250` | GAN 训练延迟启动轮数 |
| `cond_discriminator` | `True` | 是否使用条件判别器 |
| `batch_size` | `4` | 训练批次大小 |
| `num_epochs` | `1500` | 训练总轮数 |
| `learning_rate` | `2e-4` | 学习率 |
| `image_size` | `256` | 图像尺寸 |
| `cond_dim` | `256` | 条件向量维度 |
| `cond_block_type` | `"cross_attention"` | 条件注入方式：`"add"` 或 `"cross_attention"` |

## 致谢

- [DeepDRR](https://github.com/arcadelab/deepdrr) — 用于从 CT 体数据生成数字重建放射影像（DRR）
- [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239)
- [Denoising Diffusion Implicit Models (DDIM)](https://arxiv.org/abs/2010.02502)

## License

本项目仅用于学术研究目的。
