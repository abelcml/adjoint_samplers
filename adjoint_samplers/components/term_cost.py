# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import adjoint_samplers.utils.graph_utils as graph_utils


class GradEnergy:
    """ Compute ∇E(X_1)
    """
    # max_grad_E_norm:∇E 每個樣本的 L2 範數上限,由 experiment config 傳入;None 代表不裁。
    def __init__(self, energy, max_grad_E_norm = None, **kwargs):
        self.energy = energy
        self.max_grad_E_norm = max_grad_E_norm

    def clip(self, grad_E):
        # 如果設置了 max_grad_E_norm，則進行梯度裁剪
        if self.max_grad_E_norm is not None:
            # 計算梯度的範數（L2 範數），沿著最後一個維度進行計算
            norm = torch.linalg.vector_norm(grad_E, dim=-1).detach()
            # 計算裁剪係數，將 max_grad_E_norm 除以梯度範數，並限制最大值為 1
            # 係數 = min(1, 上限 / 範數):範數 ≤ 上限的樣本係數為 1、完全不變;
            # 超過的樣本被縮到長度剛好等於上限,方向不變。1e-6 只是避免範數為 0 時除以零。
            clip_coefficient = torch.clamp(self.max_grad_E_norm / (norm + 1e-6), max=1)
            # 將裁剪係數的形狀擴展到與梯度相同的維度
            clip_coefficient = clip_coefficient.unsqueeze(-1)
        else:
            # 如果未設置 max_grad_E_norm，則裁剪係數為全 1（不進行裁剪）
            clip_coefficient = torch.ones_like(grad_E)

        # 返回裁剪後的梯度，將原始梯度乘以裁剪係數
        return grad_E * clip_coefficient

    def grad_E(self, x1):
        grad_E = self.energy(x1)["forces"]
        return self.clip(grad_E)

    def __call__(self, x1):
        return self.grad_E(x1)


# For AS.
class ScoreGradTermCost(GradEnergy):
    """ Compute (∇E + ∇log p^base_1)(X_1)
    """
    def __init__(self, source, ref_sde, energy, **kwargs):
        super().__init__(energy, **kwargs)

        self._check_source_class(source)

        # Compute p^base_1 = N(x1; μ1, Σ1)
        mu0, var0 = source.loc, source.scale**2
        t1 = torch.ones(1, device=mu0.device)
        mu1, var1 = ref_sde._pt_gauss_param(t1, mu0, var0)
        self.mu1 = mu1.reshape(-1)
        self.var1 = var1.reshape(-1)

    def _check_source_class(self, source):
        from adjoint_samplers.utils.dist_utils import Delta, Gauss
        assert isinstance(source, (Delta, Gauss))

    def __call__(self, x1):
        # Compute ∇log p^base_1(x) = (μ1 - x) / Σ1
        score = (self.mu1.to(x1) - x1) / self.var1.to(x1)
        return self.grad_E(x1) + score


# For ASBS.
class CorrectorGradTermCost(GradEnergy):
    """ Compute (∇E + ∇log h)(X_1), where h is the corrector of ASBS.
    """
    def __init__(self, corrector, energy, **kwargs):
        super().__init__(energy, **kwargs)
        self.corrector = corrector

    def __call__(self, x1, **kwargs):
        t1 = torch.ones(x1.shape[0], 1).to(x1)
        with torch.no_grad():
            corrector = self.corrector(t1, x1)
        return self.grad_E(x1) + corrector


# For ASBS on n-particle systems.
class GraphCorrectorGradTermCost(CorrectorGradTermCost):
    def __init__(self, corrector, energy, **kwargs):
        super().__init__(corrector, energy, **kwargs)
        self.n_particles = energy.n_particles
        self.n_spatial_dim = energy.n_spatial_dim

    def grad_E(self, x1):
        N, D = self.n_particles, self.n_spatial_dim

        grad_E = self.energy(x1)["forces"]

        # clip spatial dim
        # 先 reshape 成 (B, N, D),所以範數是對每個粒子的 D 維分別算:
        # 上限作用在單一粒子的力上,而不是整個 N*D 向量。
        grad_E = self.clip(grad_E.view(-1, N, D)).view(-1, N * D)

        grad_E = graph_utils.remove_mean(grad_E, N, D)
        return grad_E


# For AS on n-particle systems.
class GraphScoreGradTermCost(ScoreGradTermCost):
    """ Compute (∇E + ∇log p^base_1)(X_1) for n-particle systems
    """
    def __init__(self, source, ref_sde, energy, **kwargs):
        super().__init__(source, ref_sde, energy, **kwargs)
        self.n_particles = energy.n_particles
        self.n_spatial_dim = energy.n_spatial_dim

    def _check_source_class(self, source):
        from adjoint_samplers.utils.dist_utils import Delta, CenteredParticlesGauss
        assert isinstance(source, (Delta, CenteredParticlesGauss))

    def grad_E(self, x1):
        N, D = self.n_particles, self.n_spatial_dim

        grad_E = self.energy(x1)["forces"]

        # clip spatial dim
        # 先 reshape 成 (B, N, D),所以範數是對每個粒子的 D 維分別算:
        # 上限作用在單一粒子的力上,而不是整個 N*D 向量。
        grad_E = self.clip(grad_E.view(-1, N, D)).view(-1, N * D)

        grad_E = graph_utils.remove_mean(grad_E, N, D)
        return grad_E
