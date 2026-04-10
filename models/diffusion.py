import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


# ============================================================
# 工具函数
# ============================================================

def make_beta_schedule(schedule="linear", n_timestep=1000,
                       linear_start=1e-4, linear_end=2e-2):
    """生成噪声调度的 beta 序列"""
    if schedule == "linear":
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        # cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        steps = n_timestep + 1
        x = np.linspace(0, n_timestep, steps)
        alphas_cumprod = np.cos(((x / n_timestep) + 0.008) / 1.008 * np.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = np.clip(betas, a_min=0, a_max=0.999)
    else:
        raise NotImplementedError(f"Unknown schedule: {schedule}")
    return betas


def extract_into_tensor(a, t, x_shape):
    """
    从数组 a 中按时间步 t 提取值，并reshape到与x_shape匹配的形状
    
    例如: a=[...1000个值...], t=[0,3,5](batch), x_shape=(B,C,H,W)
    返回: shape为(B,1,1,1)的tensor，方便广播
    """
    b = t.shape[0]  # batch size
    # 按照 t 中的索引从 a 中取值
    out = a.gather(-1, t)
    # reshape 为 (B, 1, 1, ...) 方便与图像做广播乘法
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))



# ============================================================
# DiffusionWrapper: 处理不同的条件注入方式
# ============================================================

class DiffusionWrapper(nn.Module):
    """
    UNet的包装器，根据 conditioning_key 选择不同的条件注入方式：
    
    - None      : 无条件，直接输入 x
    - 'concat'  : 将条件c在通道维度拼接到x
    - 'crossattn': 将条件c作为cross-attention的context
    - 'hybrid'  : 同时使用concat和crossattn
    - 'adm'     : 将条件作为类别标签y传入
    """
    def __init__(self, unet: nn.Module, conditioning_key: str = None):
        super().__init__()
        self.diffusion_model = unet
        self.conditioning_key = conditioning_key
        assert conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm'], \
            f"Unknown conditioning_key: {conditioning_key}"

    def forward(self, x, t, c_concat=None, c_crossattn=None):
        """
        x          : 噪声图像 (B, C, H, W)
        t          : 时间步   (B,)
        c_concat   : list of tensors, 在通道维度拼接的条件
        c_crossattn: list of tensors, 用于cross-attention的条件
        """
        if self.conditioning_key is None:
            # 无条件生成
            out = self.diffusion_model(x, t)

        elif self.conditioning_key == 'concat':
            # 将条件在通道维度拼接: (B, C+cond_C, H, W)
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t)

        elif self.conditioning_key == 'crossattn':
            # 将多个条件拼接后作为cross-attention context
            cc = torch.cat(c_crossattn, dim=1)  # (B, seq_len, d)
            out = self.diffusion_model(x, t, context=cc)

        elif self.conditioning_key == 'hybrid':
            # 同时使用concat和crossattn
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, dim=1)
            out = self.diffusion_model(xc, t, context=cc)

        elif self.conditioning_key == 'adm':
            # 类别条件，作为标签y传入
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)

        return out


# ============================================================
# DDPM: 经典高斯扩散模型（在图像空间）
# ============================================================

class DDPM(nn.Module):
    """
    经典 DDPM 实现
    
    核心流程:
    前向过程 (加噪): q(x_t | x_0) = sqrt(ᾱ_t)*x_0 + sqrt(1-ᾱ_t)*ε
    反向过程 (去噪): p(x_{t-1} | x_t) 由UNet预测
    
    参数说明:
    - parameterization: "eps" 表示预测噪声ε, "x0" 表示直接预测原图
    - v_posterior: 后验方差的插值系数 (0=DDPM方差, 1=直接用β)
    """
    def __init__(
        self,
        unet: nn.Module,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        loss_type: str = "l2",
        parameterization: str = "eps",   # "eps" or "x0"
        linear_start: float = 1e-4,
        linear_end: float = 2e-2,
        v_posterior: float = 0.,          # 后验方差插值
        l_simple_weight: float = 1.,
        original_elbo_weight: float = 0., # VLB损失权重
        clip_denoised: bool = True,
        log_every_t: int = 100,
        image_size: int = 32,
        channels: int = 3,
        conditioning_key: str = None,
    ):
        super().__init__()
        assert parameterization in ["eps", "x0"]
        self.parameterization = parameterization
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.image_size = image_size
        self.channels = channels
        self.loss_type = loss_type
        self.l_simple_weight = l_simple_weight
        self.original_elbo_weight = original_elbo_weight

        # UNet 包装器（处理条件注入）
        self.model = DiffusionWrapper(unet, conditioning_key)

        # 注册噪声调度相关的 buffer
        self._register_schedule(
            beta_schedule=beta_schedule,
            timesteps=timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
        )

    def _register_schedule(self, beta_schedule, timesteps,
                           linear_start, linear_end):
        """
        预计算并注册扩散过程中需要的所有统计量
        
        关键变量:
        β_t               : 每步噪声方差 (betas)
        α_t = 1 - β_t     : 每步信号保留率
        ᾱ_t = ∏α_i        : 累积乘积 (alphas_cumprod)
        """
        # 1. 计算 beta 序列
        betas = make_beta_schedule(
            beta_schedule, timesteps,
            linear_start=linear_start,
            linear_end=linear_end
        )
        alphas = 1.0 - betas
        # 累积乘积: ᾱ_t = α_1 * α_2 * ... * α_t
        alphas_cumprod = np.cumprod(alphas, axis=0)
        # ᾱ_{t-1}，在开头补1（ᾱ_0 = 1）
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.num_timesteps = int(timesteps)

        # 辅助函数：转为 float32 tensor
        to_torch = lambda x: torch.tensor(x, dtype=torch.float32)

        # === 前向过程 q(x_t|x_0) 所需系数 ===
        # x_t = sqrt(ᾱ_t)*x_0 + sqrt(1-ᾱ_t)*ε
        self.register_buffer('betas',              to_torch(betas))
        self.register_buffer('alphas_cumprod',     to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',to_torch(alphas_cumprod_prev))
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1.0 - alphas_cumprod)))

        # === 从 x_t 和 ε 反推 x_0 所需系数 ===
        # x_0 = sqrt(1/ᾱ_t)*x_t - sqrt(1/ᾱ_t - 1)*ε
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1.0 / alphas_cumprod - 1)))

        # === 后验分布 q(x_{t-1}|x_t, x_0) 所需系数 ===
        # 后验方差: β̃_t = β_t * (1-ᾱ_{t-1}) / (1-ᾱ_t)
        # v_posterior=0 时使用 β̃_t, v_posterior=1 时使用 β_t
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # 对数方差（截断避免 log(0)）
        self.register_buffer('posterior_log_variance_clipped',
                             to_torch(np.log(np.maximum(posterior_variance, 1e-20))))

        # 后验均值系数:
        # μ̃_t = coef1 * x_0 + coef2 * x_t
        # coef1 = β_t * sqrt(ᾱ_{t-1}) / (1 - ᾱ_t)
        # coef2 = (1 - ᾱ_{t-1}) * sqrt(α_t) / (1 - ᾱ_t)
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)))

        # === VLB 损失权重 ===
        if self.parameterization == "eps":
            lvlb_weights = (
                betas ** 2 /
                (2 * posterior_variance * alphas * (1 - alphas_cumprod))
            )
        elif self.parameterization == "x0":
            lvlb_weights = (
                0.5 * np.sqrt(alphas_cumprod) / (2.0 * (1 - alphas_cumprod))
            )
        lvlb_weights[0] = lvlb_weights[1]  # 第一步特殊处理
        self.register_buffer('lvlb_weights', to_torch(lvlb_weights))

    # ----------------------------------------------------------
    # 前向扩散过程（加噪）
    # ----------------------------------------------------------

    def q_sample(self, x_start, t, noise=None):
        """
        前向过程采样: q(x_t | x_0)
        
        公式: x_t = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * ε
        
        x_start: (B, C, H, W) 原始图像
        t      : (B,)          时间步
        noise  : (B, C, H, W) 高斯噪声，None时自动生成
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        # 提取对应时间步的系数并广播到图像形状
        sqrt_alphas_t      = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_t   = extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_t * x_start + sqrt_one_minus_t * noise

    def q_mean_variance(self, x_start, t):
        """
        计算前向分布 q(x_t|x_0) 的均值和方差
        用于验证和计算VLB
        """
        mean        = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance    = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance= extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    # ----------------------------------------------------------
    # 反向去噪过程
    # ----------------------------------------------------------

    def predict_start_from_noise(self, x_t, t, noise):
        """
        由 x_t 和预测的噪声 ε 反推 x_0
        
        公式（由前向过程逆推）:
        x_0 = sqrt(1/ᾱ_t) * x_t - sqrt(1/ᾱ_t - 1) * ε
        """
        coef1 = extract_into_tensor(self.sqrt_recip_alphas_cumprod,  t, x_t.shape)
        coef2 = extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return coef1 * x_t - coef2 * noise

    def q_posterior(self, x_start, x_t, t):
        """
        计算后验分布 q(x_{t-1} | x_t, x_0) 的均值和方差
        
        后验均值: μ̃_t = coef1*x_0 + coef2*x_t
        后验方差: β̃_t（预先计算好的）
        """
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(
            self.posterior_variance, t, x_t.shape)
        posterior_log_variance = extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape)

        return posterior_mean, posterior_variance, posterior_log_variance

    def p_mean_variance(self, x, t, clip_denoised=True):
        """
        计算反向过程 p(x_{t-1}|x_t) 的均值和方差
        
        步骤:
        1. UNet预测噪声（或x_0）
        2. 由预测结果得到 x_0
        3. 由 x_0 和 x_t 计算后验分布参数
        """
        # UNet 前向推理
        model_out = self.model(x, t)

        # 根据参数化方式得到 x_0 的预测
        if self.parameterization == "eps":
            # 模型预测噪声，从噪声反推 x_0
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            # 模型直接预测 x_0
            x_recon = model_out

        # 可选：将 x_0 裁剪到 [-1, 1]（稳定训练）
        if clip_denoised:
            x_recon = x_recon.clamp(-1.0, 1.0)

        # 计算后验分布参数
        model_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior(x_start=x_recon, x_t=x, t=t)

        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t):
        """
        单步去噪采样: 从 x_t 采样 x_{t-1}
        
        x_{t-1} = μ_θ(x_t, t) + σ_t * z
        其中 z~N(0,I)，当 t=0 时不添加噪声
        """
        b = x.shape[0]
        device = x.device

        # 计算均值和对数方差
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=self.clip_denoised
        )

        # 采样随机噪声
        noise = torch.randn_like(x)

        # t=0 时不加噪声（最后一步直接输出均值）
        # nonzero_mask shape: (B, 1, 1, 1) 方便广播
        nonzero_mask = (1 - (t == 0).float()).reshape(
            b, *([1] * (len(x.shape) - 1))
        )

        # x_{t-1} = μ + σ * z = μ + exp(0.5 * log_var) * z
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        """
        完整的采样循环: 从纯噪声 x_T 逐步去噪到 x_0
        
        shape: (B, C, H, W)
        """
        device = self.betas.device
        b = shape[0]

        # 从标准高斯分布开始
        img = torch.randn(shape, device=device)
        intermediates = [img]

        # 从 T-1 倒数到 0
        for i in tqdm(reversed(range(self.num_timesteps)),
                      desc='DDPM Sampling', total=self.num_timesteps):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t)

            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=4, return_intermediates=False):
        """生成样本的入口函数"""
        shape = (batch_size, self.channels, self.image_size, self.image_size)
        return self.p_sample_loop(shape, return_intermediates=return_intermediates)

    # ----------------------------------------------------------
    # 损失计算
    # ----------------------------------------------------------

    def get_loss(self, pred, target, mean=True):
        """计算 L1 或 L2 损失"""
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
        elif self.loss_type == 'l2':
            loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError(f"Unknown loss type: {self.loss_type}")
        return loss.mean() if mean else loss

    def p_losses(self, x_start, t, noise=None):
        """
        计算训练损失
        
        损失由两部分组成:
        1. loss_simple: 简单均方误差损失（主要损失）
        2. loss_vlb   : 变分下界损失（辅助，通常权重很小）
        """
        # 生成噪声并加噪
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # UNet 预测
        model_out = self.model(x_noisy, t)

        # 确定预测目标
        if self.parameterization == "eps":
            target = noise    # 预测噪声
        elif self.parameterization == "x0":
            target = x_start  # 直接预测原图

        # 逐样本计算损失（不立即平均，用于 VLB 加权）
        loss_per_sample = self.get_loss(model_out, target, mean=False)
        loss_per_sample = loss_per_sample.mean(dim=[1, 2, 3])  # (B,)

        # 简单损失
        loss_simple = loss_per_sample.mean() * self.l_simple_weight

        # VLB 损失（使用预先计算的重要性权重）
        loss_vlb = (self.lvlb_weights[t] * loss_per_sample).mean()

        # 总损失
        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict = {
            'loss_simple': loss_simple.item(),
            'loss_vlb':    loss_vlb.item(),
            'loss':        loss.item(),
        }
        return loss, loss_dict

    def forward(self, x):
        """
        训练时的前向传播
        随机采样时间步 t，计算损失
        """
        t = torch.randint(0, self.num_timesteps, (x.shape[0],),
                          device=x.device).long()
        return self.p_losses(x, t)


# ============================================================
# LatentDiffusion: 在潜在空间做扩散（LDM）
# ============================================================

class LatentDiffusion(DDPM):
    """
    潜在扩散模型 (LDM)
    
    核心思路:
    1. 使用预训练的 VAE 将图像编码到低维潜在空间 z
    2. 在潜在空间 z 上运行 DDPM
    3. 解码时将 z 解码回像素空间
    
    优势: 在低维潜在空间运算，大幅降低计算量
    """
    def __init__(
        self,
        unet: nn.Module,
        first_stage_model: nn.Module,   # VAE 编码器/解码器
        cond_stage_model: nn.Module,     # 条件编码器（如CLIP文本编码器）
        scale_factor: float = 1.0,       # 潜在空间缩放因子
        cond_stage_trainable: bool = False,
        conditioning_key: str = None,
        **ddpm_kwargs
    ):
        super().__init__(unet=unet, conditioning_key=conditioning_key, **ddpm_kwargs)

        # VAE（冻结，不参与训练）
        self.first_stage_model = first_stage_model.eval()
        for p in self.first_stage_model.parameters():
            p.requires_grad = False

        # 条件编码器（可选择是否参与训练）
        self.cond_stage_model = cond_stage_model
        self.cond_stage_trainable = cond_stage_trainable
        if not cond_stage_trainable:
            self.cond_stage_model.eval()
            for p in self.cond_stage_model.parameters():
                p.requires_grad = False

        # 潜在空间缩放因子（归一化潜在表示）
        self.scale_factor = scale_factor

    # ----------------------------------------------------------
    # 编码/解码
    # ----------------------------------------------------------

    @torch.no_grad()
    def encode_first_stage(self, x):
        """
        将图像编码到潜在空间
        x: (B, C, H, W) 像素图像
        返回: 潜在分布或张量
        """
        return self.first_stage_model.encode(x)

    def get_first_stage_encoding(self, encoder_posterior):
        """
        从编码器输出中采样潜在向量 z，并缩放
        
        若 VAE 输出高斯分布参数 → 从中采样
        若已经是 Tensor → 直接使用
        """
        if hasattr(encoder_posterior, 'sample'):
            # VAE 输出的是高斯分布，进行重参数化采样
            z = encoder_posterior.sample()
        else:
            z = encoder_posterior
        return self.scale_factor * z

    @torch.no_grad()
    def decode_first_stage(self, z):
        """
        将潜在向量解码回像素图像
        z: (B, C_latent, H_latent, W_latent)
        """
        z = (1.0 / self.scale_factor) * z
        return self.first_stage_model.decode(z)

    # ----------------------------------------------------------
    # 条件编码
    # ----------------------------------------------------------

    def get_learned_conditioning(self, c):
        """
        使用条件编码器对条件 c 进行编码
        c 可以是文本 token、图像等
        """
        if hasattr(self.cond_stage_model, 'encode'):
            # 支持 encode 方法（如 CLIP）
            c_encoded = self.cond_stage_model.encode(c)
            # 若输出是分布则取众数（确定性编码）
            if hasattr(c_encoded, 'mode'):
                c_encoded = c_encoded.mode()
        else:
            c_encoded = self.cond_stage_model(c)
        return c_encoded

    # ----------------------------------------------------------
    # 覆盖基类方法：在潜在空间操作
    # ----------------------------------------------------------

    def apply_model(self, x_noisy, t, cond):
        """
        将条件整理成 DiffusionWrapper 需要的格式并调用 UNet
        
        cond 可以是:
        - Tensor: 转换为列表
        - list  : 直接使用
        - dict  : 直接透传（已经是正确格式）
        """
        if isinstance(cond, dict):
            # 已经是 {c_concat: [...], c_crossattn: [...]} 格式
            pass
        else:
            # 统一转为列表格式
            if not isinstance(cond, list):
                cond = [cond]
            # 根据 conditioning_key 决定放入哪个槽
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        return self.model(x_noisy, t, **cond)

    def p_losses(self, x_start, cond, t, noise=None):
        """
        在潜在空间计算损失（覆盖基类方法）
        
        x_start: 潜在向量 z (B, C_latent, H_latent, W_latent)
        cond   : 已编码的条件
        t      : 时间步
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        # 前向加噪
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # 带条件的 UNet 预测
        model_output = self.apply_model(x_noisy, t, cond)

        # 确定目标
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start

        # 计算损失
        loss_per_sample = self.get_loss(model_output, target, mean=False)
        loss_per_sample = loss_per_sample.mean(dim=[1, 2, 3])  # (B,)

        loss_simple = loss_per_sample.mean() * self.l_simple_weight
        loss_vlb = (self.lvlb_weights[t] * loss_per_sample).mean()
        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict = {
            'loss_simple': loss_simple.item(),
            'loss_vlb':    loss_vlb.item(),
            'loss':        loss.item(),
        }
        return loss, loss_dict

    def forward(self, x, c):
        """
        训练前向传播
        
        x: 原始图像 (B, C, H, W)
        c: 原始条件（文本等，会在此处编码）
        """
        # 编码到潜在空间
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()

        # 编码条件（如果条件编码器不参与训练则跳过）
        if not self.cond_stage_trainable:
            with torch.no_grad():
                c = self.get_learned_conditioning(c)
        else:
            c = self.get_learned_conditioning(c)

        # 随机时间步
        t = torch.randint(0, self.num_timesteps, (z.shape[0],),
                          device=z.device).long()

        return self.p_losses(z, c, t)

    def p_mean_variance(self, x, c, t, clip_denoised=True):
        """
        在潜在空间计算反向过程的均值和方差（带条件）
        """
        model_out = self.apply_model(x, t, c)

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out

        if clip_denoised:
            x_recon = x_recon.clamp(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior(x_start=x_recon, x_t=x, t=t)

        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, c, t):
        """带条件的单步去噪采样"""
        b = x.shape[0]
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, c=c, t=t, clip_denoised=self.clip_denoised
        )
        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(
            b, *([1] * (len(x.shape) - 1))
        )
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False):
        """在潜在空间的完整采样循环"""
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]

        for i in tqdm(reversed(range(self.num_timesteps)),
                      desc='LDM Sampling', total=self.num_timesteps):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, cond, t)

            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, cond, batch_size=4, return_intermediates=False):
        """
        生成样本的入口
        
        1. 在潜在空间采样 z
        2. 解码 z 为像素图像
        """
        shape = (batch_size, self.channels, self.image_size, self.image_size)
        # 编码条件
        if not isinstance(cond, dict):
            c = self.get_learned_conditioning(cond)
        else:
            c = cond

        # 潜在空间采样
        samples = self.p_sample_loop(c, shape, return_intermediates=return_intermediates)

        if return_intermediates:
            z_samples, intermediates = samples
            # 解码到像素空间
            x_samples = self.decode_first_stage(z_samples)
            x_intermediates = [self.decode_first_stage(z) for z in intermediates]
            return x_samples, x_intermediates

        return self.decode_first_stage(samples)
