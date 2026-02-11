from typing import Protocol
import torch
import torch.nn.functional as F
from typing import Tuple, Callable

# 自定义一种协议类型，用于定义扩散模型的接口
class DiffusionModel(Protocol):
    def __call__(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        ...

# TODO 可能涉及到把tensor变量to GPU的手动修改步骤
class BetaScheduleCoefficients:
    def __init__(
        self,
        betas: torch.Tensor,
        alphas: torch.Tensor,
        alphas_cumprod: torch.Tensor,
        alphas_cumprod_prev: torch.Tensor,
        sqrt_alphas_cumprod: torch.Tensor,
        sqrt_one_minus_alphas_cumprod: torch.Tensor,
        log_one_minus_alphas_cumprod: torch.Tensor,
        sqrt_recip_alphas_cumprod: torch.Tensor,
        sqrt_recipm1_alphas_cumprod: torch.Tensor,
        posterior_variance: torch.Tensor,
        posterior_log_variance_clipped: torch.Tensor,
        posterior_mean_coef1: torch.Tensor,
        posterior_mean_coef2: torch.Tensor
    ):
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = sqrt_alphas_cumprod
        self.sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod
        self.log_one_minus_alphas_cumprod = log_one_minus_alphas_cumprod
        self.sqrt_recip_alphas_cumprod = sqrt_recip_alphas_cumprod
        self.sqrt_recipm1_alphas_cumprod = sqrt_recipm1_alphas_cumprod
        self.posterior_variance = posterior_variance
        self.posterior_log_variance_clipped = posterior_log_variance_clipped
        self.posterior_mean_coef1 = posterior_mean_coef1
        self.posterior_mean_coef2 = posterior_mean_coef2

    @staticmethod
    def from_beta(betas: torch.Tensor):
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.], dtype=betas.dtype, device=betas.device), alphas_cumprod[:-1]]
        )

        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod)
        sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_log_variance_clipped = torch.log(torch.clamp(posterior_variance, min=1e-20))
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

        return BetaScheduleCoefficients(
            betas, alphas, alphas_cumprod, alphas_cumprod_prev,
            sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, log_one_minus_alphas_cumprod,
            sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod,
            posterior_variance, posterior_log_variance_clipped,
            posterior_mean_coef1, posterior_mean_coef2
        )

    @staticmethod
    def vp_beta_schedule(timesteps: int) -> torch.Tensor:
        t = torch.arange(1, timesteps + 1, dtype=torch.float32)
        T = float(timesteps)
        b_max = 10.0
        b_min = 0.1
        alpha = torch.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T**2)
        betas = 1 - alpha
        return betas

    @staticmethod
    def cosine_beta_schedule(timesteps: int) -> torch.Tensor:
        s = 0.008
        t = torch.linspace(0, 1, timesteps + 1)
        alphas_cumprod = torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = torch.clamp(betas, max=0.999)
        return betas


class GaussianDiffusion:
    def __init__(self, num_timesteps: int, device: torch.device = torch.device('cpu')):
        self.num_timesteps = num_timesteps
        self.device = device
        self.B = self._beta_schedule()

    def _beta_schedule(self):
        betas = BetaScheduleCoefficients.cosine_beta_schedule(self.num_timesteps)
        return BetaScheduleCoefficients.from_beta(betas)

    def p_mean_variance(self, t: int, x: torch.Tensor, noise_pred: torch.Tensor):
        B = self.B
        x_recon = x * B.sqrt_recip_alphas_cumprod[t] - noise_pred * B.sqrt_recipm1_alphas_cumprod[t]
        x_recon = torch.clamp(x_recon, -1., 1.)
        model_mean = x_recon * B.posterior_mean_coef1[t] + x * B.posterior_mean_coef2[t]
        model_log_variance = B.posterior_log_variance_clipped[t]
        return model_mean, model_log_variance

    def p_sample(self, model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
                shape: Tuple[int], device=None) -> torch.Tensor:
        device = device or self.device
        x = torch.randn(shape, device=device)  # 初始高斯噪声
        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((shape[0],), t, dtype=torch.long, device=device)
            # 注意：x 是前一步生成的，需要 requires_grad_ 来追踪路径
            x.requires_grad_(True)
            noise_pred = model(t_tensor, x)  # 可导模型输出
            model_mean, model_log_variance = self.p_mean_variance(t, x, noise_pred)
            if t > 0:
                noise = torch.randn_like(x)  # 动态采样，避免预先常量采样导致断图
                x = model_mean + torch.exp(0.5 * model_log_variance) * noise  # 可导路径
            else:
                x = model_mean  # 最后一步不加 noise
        return x  # ✅ 可用于反向传播

    def q_sample(self, t: int, x_start: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        B = self.B
        return B.sqrt_alphas_cumprod[t] * x_start + B.sqrt_one_minus_alphas_cumprod[t] * noise

    def p_loss(self, model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
               t: torch.Tensor, x_start: torch.Tensor) -> torch.Tensor:
        assert t.ndim == 1 and t.shape[0] == x_start.shape[0]
        noise = torch.randn_like(x_start)
        x_noisy = torch.stack([self.q_sample(t[i], x_start[i], noise[i]) for i in range(len(t))])
        noise_pred = model(t, x_noisy)
        loss = F.mse_loss(noise_pred, noise)
        return loss

    def weighted_p_loss(self, weights: torch.Tensor,
                        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                        t: torch.Tensor, x_start: torch.Tensor) -> torch.Tensor:
        if weights.dim() == 1:
            weights = weights.view(-1, 1)
        assert t.ndim == 1 and t.shape[0] == x_start.shape[0]
        noise = torch.randn_like(x_start)
        x_noisy = torch.stack([self.q_sample(t[i], x_start[i], noise[i]) for i in range(len(t))])
        noise_pred = model(t, x_noisy)
        loss = weights * (noise_pred - noise) ** 2
        return loss.mean()
