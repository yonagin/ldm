import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

from modules.encoder import Encoder
from modules.decoder import Decoder


# ============================================================================
# SoftRank Implementations
# ============================================================================

class SoftRankSinkhorn(nn.Module):
    """使用Sinkhorn算法实现的SoftRank"""
    def __init__(self, tau=1.0, sinkhorn_iters=20):
        super().__init__()
        self.tau_param = nn.Parameter(torch.tensor(tau))
        self.sinkhorn_iters = sinkhorn_iters
    
    @property
    def tau(self):
        return F.softplus(self.tau_param) + 0.01
    
    def log_sinkhorn(self, log_alpha, n_iters):
        """Sinkhorn归一化: 将log空间矩阵归一化为双随机矩阵"""
        for _ in range(n_iters):
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
        return torch.exp(log_alpha)
    
    def compute_soft_permutation(self, z):
        """
        z: [B, n] 输入值
        返回: [B, n, n] soft permutation matrix
        """
        batch_size, n = z.shape
        device = z.device
        dtype = z.dtype
    
        # 线性归一化到 [0, 1]，保持区分度
        z_min = z.min(dim=-1, keepdim=True).values
        z_max = z.max(dim=-1, keepdim=True).values
        z_processed = (z - z_min) / (z_max - z_min + 1e-8)
    
        # 锚点避开边界
        margin = 0.5 / n
        y = torch.linspace(margin, 1.0 - margin, steps=n, device=device, dtype=dtype)
        y = y.view(1, 1, n)          # [1, 1, n]
        z_expand = z_processed.unsqueeze(-1)  # [B, n, 1]
        
        cost = (z_expand - y) ** 2    # [B, n, n]
        log_alpha = -cost / self.tau
        
        P_soft = self.log_sinkhorn(log_alpha, self.sinkhorn_iters)
        return P_soft
    
    def forward(self, x):
        """
        x: [B, n] 输入值
        返回: [B, n] soft rank values (在1到n之间)
        如果return_permutation=True，还返回permutation matrix
        """
        P_soft = self.compute_soft_permutation(x)  # [B, n, n]
        n = x.shape[1]
        # 目标排序位置 [1, 2, 3, ..., n]
        rank_values = torch.arange(1, n + 1, dtype=x.dtype, device=x.device).view(1, n)
        # P_soft @ rank_values 得到软排序
        soft_ranks = torch.matmul(P_soft, rank_values.unsqueeze(-1)).squeeze(-1)  # [B, n]
        
        return soft_ranks, P_soft


class SoftRankSigmoid(nn.Module):
    """使用Sigmoid方法实现的SoftRank
    rank_i = 1 + sum_{j != i} sigmoid(alpha*(x_i - x_j))
    """
    def __init__(self, alpha=10.0):
        super().__init__()
        self.alpha_param = nn.Parameter(torch.tensor(alpha))
    
    @property
    def alpha(self):
        return F.softplus(self.alpha_param)
    
    def forward(self, x):
        """
        x: [B, n] 输入值
        返回: [B, n] soft rank values
        注意：sigmoid方法不返回permutation matrix
        """
        B, n = x.shape
        # x_i - x_j for all pairs
        x_expand_i = x.unsqueeze(2)  # [B, n, 1]
        x_expand_j = x.unsqueeze(1)  # [B, 1, n]
        diff = x_expand_i - x_expand_j  # [B, n, n]
        
        # sigmoid(alpha * (x_i - x_j))
        soft_comp = torch.sigmoid(self.alpha * diff)  # [B, n, n]
        
        # 对角线需要排除（j != i），所以对角线设为0
        mask = torch.eye(n, device=x.device, dtype=x.dtype).unsqueeze(0)  # [1, n, n]
        soft_comp = soft_comp * (1 - mask)
        
        # sum over j
        soft_ranks = 1 + soft_comp.sum(dim=2)  # [B, n]
        
        return soft_ranks, None 

# ============================================================================
# Rank Layer Implementations
# ============================================================================

class RankLayer(nn.Module):
    """Args:
        slice_mode: "channel" 或 "spatial"
            - "channel": 每个空间位置的通道值独立排序
            - "spatial": 每个patch内的空间位置独立排序
        softrank_method: "sinkhorn" 或 "sigmoid"
        gradient_mode: "soft" 或 "STE"
            - "soft": 直接使用softrank的梯度
            - "STE": 前向用hardrank，反向用softrank
        normalize_range: "0_1" 或 "-1_1"
        use_ema_basis: 是否使用EMA估计的基础秩向量
        ema_decay: EMA更新的衰减率
        ema_interpolate: 使用插值还是矩阵乘法 ("interpolate" 或 "matmul")
        patch_size: 当slice_mode="spatial"时使用
    """
    def __init__(
        self,
        slice_mode: Literal["channel", "spatial"] = "channel",
        softrank_method: Literal["sinkhorn", "sigmoid"] = "sinkhorn",
        gradient_mode: Literal["soft", "STE"] = "STE",
        normalize_range: Literal["0_1", "-1_1"] = "0_1",
        use_ema_basis: bool = False,
        ema_decay: float = 0.99,
        ema_interpolate: Literal["interpolate", "matmul"] = "interpolate",
        patch_size: int = 4,
        latent_dim: int = 16,  # 用于初始化basis vector
        tau: float = 1.0,
        alpha: float = 10.0,
        sinkhorn_iters: int = 20
    ):
        super().__init__()
        self.slice_mode = slice_mode
        self.softrank_method = softrank_method
        self.gradient_mode = gradient_mode
        self.normalize_range = normalize_range
        self.use_ema_basis = use_ema_basis
        self.ema_decay = ema_decay
        self.ema_interpolate = ema_interpolate
        self.patch_size = patch_size
        
        # 初始化SoftRank模块
        if softrank_method == "sinkhorn":
            self.softrank = SoftRankSinkhorn(tau=tau, sinkhorn_iters=sinkhorn_iters)
        else:
            self.softrank = SoftRankSigmoid(alpha=alpha)
        
        # 初始化EMA basis vector
        if use_ema_basis:
            # 根据slice_mode确定basis维度
            if slice_mode == "channel":
                n_dim = latent_dim  # 通道数
            else:  # spatial
                n_dim = patch_size ** 2  # 每个patch的空间位置数
            
            # 初始化为均匀分布在归一化范围内
            if normalize_range == "0_1":
                init_v = torch.linspace(0.0, 1.0, n_dim)
            else:  # -1_1
                init_v = torch.linspace(-1.0, 1.0, n_dim)
            
            self.register_buffer('v', init_v.clone())
            self.register_buffer('ema_count', torch.tensor(0))
    
    def normalize_ranks(self, ranks):
        """将rank值从[1, max_rank]归一化到指定范围"""
        # 先归一化到 [0, 1]
        max_rank = ranks.shape[-1]
        normalized = (ranks - 0.5) / max_rank
        
        if self.normalize_range == "-1_1":
            normalized = normalized * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        
        return normalized
    
    @torch.no_grad()
    def update_ema(self, z_flat):
        """
        使用 EMA 更新基础秩向量 v
        
        Args:
            z_flat: [B, n] 排序前的 z 值
        
        更新规则:
            v[k] = decay * v[k] + (1 - decay) * mean(z_sorted[:, k])
        """
        # 对每个样本排序
        z_sorted, _ = torch.sort(z_flat, dim=-1)  # [B, n]
        
        # 计算每个位置的均值
        z_mean = z_sorted.mean(dim=0)  # [n]
        
        # EMA更新
        self.v.mul_(self.ema_decay).add_(z_mean, alpha=1 - self.ema_decay)
        self.ema_count.add_(1)
    
    def interpolate_v(self, v, soft_ranks):
        """
        使用插值方法根据soft_ranks从基础向量v中获取值
        
        Args:
            v: [n] 基础秩向量
            soft_ranks: [B, n] soft rank values (在[1, n]之间)
        
        Returns:
            [B, n] 插值后的值
        """
        B, n = soft_ranks.shape
        device = soft_ranks.device
        
        # 将soft_ranks转换为索引 (从[1, n]转为[0, n-1])
        indices = soft_ranks - 1.0  # [B, n]
        
        # 计算左右索引
        indices_floor = torch.floor(indices).long().clamp(0, n - 2)  # [B, n]
        indices_ceil = indices_floor + 1  # [B, n]
        
        # 计算插值权重
        weights = indices - indices_floor.float()  # [B, n]
        
        # 获取左右值
        v_floor = v[indices_floor]  # [B, n]
        v_ceil = v[indices_ceil]  # [B, n]
        
        # 线性插值
        interpolated = v_floor * (1 - weights) + v_ceil * weights  # [B, n]
        
        return interpolated
    
    def apply_basis_transform(self, z_flat, soft_ranks, P_soft=None):
        """
        基于EMA basis vector进行变换
        
        Args:
            z_flat: [B, n] 原始值（用于EMA更新）
            soft_ranks: [B, n] soft rank values
            P_soft: [B, n, n] permutation matrix (仅sinkhorn方法可用)
        
        Returns:
            [B, n] 变换后的值
        """
        # 更新EMA
        if self.training:
            self.update_ema(z_flat)
        
        # 获取当前的basis vector
        v_current = self.v
        
        if self.ema_interpolate == "matmul" and P_soft is not None:
            # 使用矩阵乘法: P_soft @ v_current
            # P_soft: [B, n, n], v_current: [n]
            z_transformed = torch.matmul(P_soft, v_current.unsqueeze(-1)).squeeze(-1)  # [B, n]
        else:
            # 使用插值方法
            z_transformed = self.interpolate_v(v_current, soft_ranks)  # [B, n]
        
        return z_transformed
    
    def _soft_rank(self, z_flat, normalize=True):
        """
        核心排序逻辑：输入平铺后的 Tensor [N, D]，输出排序后的 Tensor [N, D]
        N: Batch * Groups
        D: Ranking Dimension (Channel or Spatial_Size)
        """
        soft_ranks, P_soft = self.softrank(z_flat)

        if normalize:
            if self.use_ema_basis:
                norm_soft = self.apply_basis_transform(z_flat, soft_ranks, P_soft)
            else:
                norm_soft = self.normalize_ranks(soft_ranks)

            return norm_soft
        
        return soft_ranks
    
    def hard_rank(self, x, normalize=True):
        """计算硬排序的rank值"""
        # argsort两次得到rank (从小到大，rank从1开始)
        hard_ranks = torch.argsort(torch.argsort(x, dim=-1), dim=-1).float() + 1

        if normalize:
            if self.use_ema_basis:
                norm_hard = self.interpolate_v(self.v, hard_ranks)
            else:
                norm_hard = self.normalize_ranks(hard_ranks)
                
            return norm_hard
        
        return hard_ranks
    
    def flatten(self, z: torch.Tensor):
        """
        将 [B, C, H, W] 展平为 [N, D]
        
        channel 模式: [B, C, H, W] -> [B*H*W, C]
        spatial 模式: [B, C, H, W] -> [B*C*hp*wp, p*p]
        
        返回: (z_flat, shape_info) — shape_info 用于逆向恢复
        """
        B, C, H, W = z.shape
        p = self.patch_size
        h_patches, w_patches = H // p, W // p

        # 保存恢复所需的形状信息
        shape_info = {
            "B": B, "C": C, "H": H, "W": W,
            "h_patches": h_patches, "w_patches": w_patches, "p": p
        }

        if self.slice_mode == "channel":
            # [B, C, H, W] -> [B, H, W, C] -> [B*H*W, C]
            z_flat = z.permute(0, 2, 3, 1).reshape(-1, C)
        else:
            # unfold 沿 H 和 W 两个维度提取 patch
            # [B, C, H, W] -> [B, C, hp, wp, p, p] -> [B*C*hp*wp, p*p]
            z_flat = (
                z.unfold(2, p, p)   # [B, C, hp, W, p]
                 .unfold(3, p, p)   # [B, C, hp, wp, p, p]
                 .reshape(-1, p * p)
            )

        return z_flat, shape_info

    def unflatten(self, z_flat: torch.Tensor, shape_info: dict):
        """
        将 [N, D] 恢复为 [B, C, H, W]
        
        channel 模式: [B*H*W, C] -> [B, C, H, W]
        spatial 模式: [B*C*hp*wp, p*p] -> [B, C, H, W]
        """
        B = shape_info["B"]
        C = shape_info["C"]
        H = shape_info["H"]
        W = shape_info["W"]
        h_patches = shape_info["h_patches"]
        w_patches = shape_info["w_patches"]
        p = shape_info["p"]

        if self.slice_mode == "channel":
            # [B*H*W, C] -> [B, H, W, C] -> [B, C, H, W]
            z = z_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
        else:  # spatial
            # [B*C*hp*wp, p*p] -> [B, C, hp, wp, p, p]
            z_patches = z_flat.reshape(B, C, h_patches, w_patches, p, p)
            # [B, C, hp, wp, p, p] -> [B, C, hp, p, wp, p]
            z_patches = z_patches.permute(0, 1, 2, 4, 3, 5)
            # [B, C, hp, p, wp, p] -> [B, C, H, W]
            z = z_patches.reshape(B, C, H, W)

        return z

    def rank(self, z, return_hard=True, normalize=True):
        """
        z: [B, C, H, W]
        """
        z_flat, info = self.flatten(z)
        
        if return_hard:
            ranked_flat = self.hard_rank(z_flat,normalize)

        else:
            ranked_flat = self._soft_rank(z_flat,normalize)

        return self.unflatten(ranked_flat, info)
        
    def forward(self, z):
        """
        z: [B, C, H, W]
        """
        z_flat, info = self.flatten(z)

        norm_ranks_soft = self._soft_rank(z_flat)
        
        if self.gradient_mode == "STE":
            norm_ranks_hard = self.hard_rank(z_flat)
            ranked_flat = norm_ranks_soft + (norm_ranks_hard - norm_ranks_soft).detach()

        else:
            ranked_flat = norm_ranks_soft

        return self.unflatten(ranked_flat, info)


# ============================================================================
# RankAE Model
# ============================================================================

class RankAE(nn.Module):
    def __init__(
        self,
        in_channels,
        latent_dim,
        slice_mode="channel",
        softrank_method="sigmoid",
        loss_type: str = "l2",
        gradient_mode="STE",
        normalize_range="-1_1",
        use_ema_basis=False,
        ema_decay=0.99,
        ema_interpolate="interpolate",
        patch_size=4,
        tau=1.0,
        alpha=10.0,
        sinkhorn_iters=20
    ):
        super().__init__()
        self.use_ema_basis = use_ema_basis
        self.softrank_method = softrank_method
        self.encoder = Encoder(in_channels, latent_dim)
        self.rank_layer = RankLayer(
            slice_mode=slice_mode,
            softrank_method=softrank_method,
            gradient_mode=gradient_mode,
            normalize_range=normalize_range,
            use_ema_basis=use_ema_basis,
            ema_decay=ema_decay,
            ema_interpolate=ema_interpolate,
            patch_size=patch_size,
            latent_dim=latent_dim,
            tau=tau,
            alpha=alpha,
            sinkhorn_iters=sinkhorn_iters
        )
        self.decoder = Decoder(in_channels, latent_dim)
        self.pre_norm = nn.LayerNorm(latent_dim, elementwise_affine=False)
        self.loss_type = loss_type

    def encode(self, x):
        z = self.encoder(x)
        return z
    
    def decode(self, z, return_hard=True, normalize=True):
        if self.use_ema_basis and not self.softrank_method == "sigmoid":
            # [B, C, H, W] -> [B, H, W, C] 以便做 LayerNorm
            z_perm = z.permute(0, 2, 3, 1)
            z_norm = self.pre_norm(z_perm)
            z = z_norm.permute(0, 3, 1, 2) # 转回 [B, C, H, W]
        # Rank (the key operation)
        ranked_z = self.rank_layer.rank(z, return_hard=return_hard, normalize=normalize)

        return self.decoder(ranked_z)
    
    def get_loss(self, pred, target, mean=True):
        """计算 L1 或 L2 损失"""
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
        elif self.loss_type == 'l2':
            loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError(f"Unknown loss type: {self.loss_type}")
        return loss.mean() if mean else loss

    def forward(self, x):
        # Encode
        z = self.encoder(x)
        if self.use_ema_basis and not self.softrank_method == "sigmoid":
            # [B, C, H, W] -> [B, H, W, C] 以便做 LayerNorm
            z_perm = z.permute(0, 2, 3, 1)
            z_norm = self.pre_norm(z_perm)
            z = z_norm.permute(0, 3, 1, 2) # 转回 [B, C, H, W]
        # Rank (the key operation)
        ranked_z = self.rank_layer(z)
        
        # Decode
        x_recon = self.decoder(ranked_z)
        
        return self.get_loss(x_recon, x)
