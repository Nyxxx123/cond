"""
EMA (Exponential Moving Average) 工具
用于稳定训练，提升采样质量
"""

import copy
import torch


class EMA:
    """
    指数移动平均
    """
    def __init__(self, model, decay=0.9999, device=None):
        """
        Args:
            model: 原始模型
            decay: 衰减率，越接近1越平滑，推荐0.999-0.9999
            device: 存储设备
        """
        self.decay = decay
        self.device = device if device else next(model.parameters()).device

        # 深拷贝模型参数
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()

        # 将所有参数移动到指定设备
        for param in self.shadow.parameters():
            param.requires_grad_(False)
            param.data = param.data.to(self.device)

        self.n_averaged = 0

    def update(self, model):
        """
        更新EMA参数（每步训练后调用）
        """
        self.n_averaged += 1

        # 使用衰减率（前几步用较小衰减，加速收敛）
        decay = min(self.decay, (1 + self.n_averaged) / (10 + self.n_averaged))

        with torch.no_grad():
            for ema_param, model_param in zip(self.shadow.parameters(), model.parameters()):
                ema_param.data = decay * ema_param.data + (1 - decay) * model_param.data

    def apply_to(self, model):
        """
        将EMA权重复制到模型中（用于采样）
        """
        with torch.no_grad():
            for ema_param, model_param in zip(self.shadow.parameters(), model.parameters()):
                model_param.data.copy_(ema_param.data)

    def state_dict(self):
        """保存EMA状态"""
        return {
            'shadow': self.shadow.state_dict(),
            'decay': self.decay,
            'n_averaged': self.n_averaged
        }

    def load_state_dict(self, state_dict):
        """加载EMA状态"""
        self.shadow.load_state_dict(state_dict['shadow'])
        self.decay = state_dict['decay']
        self.n_averaged = state_dict['n_averaged']