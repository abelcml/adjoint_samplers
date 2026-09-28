# Copyright (c) Meta Platforms, Inc. and affiliates.

import math
import torch
from torch import distributions


########################################
######### Target Distributions #########
########################################

class GMM1D(distributions.Distribution): #我現在定義一個新的機率分佈，請把它當成 PyTorch 的分佈來用。不是隨便寫 class , 是「按照 PyTorch 規定」做一個分佈物件
    """ A simple bi-modal Gaussian mixtures in 1D for demo purposes
    """
    def __init__(self, device="cpu") -> None:
        super().__init__()

        self.dim = 1
        self.name = "gmm1d"
        self._initialize_distr(device)

    def _initialize_distr(self, device) -> None: ## device: 放tensor的位置 None:不回傳 只做初始化
        loc = torch.tensor([-1, 2], device=device, dtype=torch.float).reshape(2, 1)
        scale = torch.tensor([.7, .4], device=device, dtype=torch.float).reshape(2, 1)
        weights = torch.tensor([.5, .5], device=device, dtype=torch.float).reshape(2)

        modes = distributions.Independent(
            distributions.Normal(loc, scale), 1
        )
        mix = distributions.Categorical(weights)
        self.distr = distributions.MixtureSameFamily(mix, modes)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        log_prob = self.distr.log_prob(x).unsqueeze(-1)
        assert log_prob.shape == (*x.shape[:-1], 1)
        return log_prob

    def sample(self, shape: tuple) -> torch.Tensor:
        return self.distr.sample(torch.Size(shape))

    def to(self, device) -> distributions.Distribution:
        self._initialize_distr(device)
        return self

class GMM2D(distributions.Distribution):
    arg_constraints = {}
    def __init__(self, device="cpu") -> None:
        super().__init__()

        self.dim = 2
        self.name = "gmm2d"
        self._initialize_distr(device)

    def _initialize_distr(self, device) -> None:
        n_components = 8
        radius = 4.0
        # 決定 ∇E 的大小:mode 附近 |∇E| ≈ 到該 mode 的距離 / component_std² (≈ 距離 × 11)。
        # 所以 max_grad_E_norm=100 對應離最近 mode 約 9 的距離。
        component_std = 0.30   

        angles = torch.linspace(
            0,
            2 * math.pi,
            n_components + 1,
            device=device
        )[:-1]

        ## turn angles into 2d coordinates
        loc = torch.stack( 
            [
                radius * torch.cos(angles), # [x1 , x2 ,x3 ...]
                radius * torch.sin(angles),
            ],
            dim=1,  # stack x/y into [(x1, y1), (x2, y2), ...]

        )  #loc = μk

        scale = torch.full(
        (n_components, 2),
        component_std,
        device=device,
        ) # fill tensor with σk

        weights = torch.full(
            (n_components, ),
            1.0 / n_components,
            device=device,
        ) #weights = πk

        # Normal (loc , scale) 建立component 1: Normal(x1), Normal(y1) , component 2: Normal(x2), Normal(y2) ...
        modes = distributions.Independent(
        distributions.Normal(loc, scale), 1
        )  #1 :把x,y scalar組合成 N([x,y];μk​,σ2I)

        # 先從 8 個 component 中選一個。
        mix = distributions.Categorical(weights)

        # MixtureSameFamily : 組合起來步驟 我先隨機選一個高斯峰，再從那個峰裡抽一個點先用 Categorical(weights) 選 k , 再從 Normal(loc[k], scale[k]) 生成一個點
        self.distr = distributions.MixtureSameFamily(
        mix,
        modes,
        )  


    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        log_prob = self.distr.log_prob(x).unsqueeze(-1)
        assert log_prob.shape == (*x.shape[:-1], 1)
        return log_prob

    def sample(self, shape: tuple) -> torch.Tensor:
        return self.distr.sample(torch.Size(shape))

    def to(self, device) -> distributions.Distribution:
        self._initialize_distr(device)
        return self       
########################################
######### Source Distributions #########
########################################

class Gauss(distributions.Distribution):
    def __init__(
        self,
        dim,
        loc: float = 0.0,
        scale: float = 1.0,
        device: str ="cpu"
    ) -> None:
        super().__init__()

        self.dim = dim
        self.loc = torch.tensor(loc, device=device, dtype=torch.float)
        self.scale = torch.tensor(scale, device=device, dtype=torch.float)
        self.name = "gauss"

    def sample(self, shape: tuple) -> torch.Tensor:
        z = torch.randn(*shape, self.dim, device=self.loc.device)
        return z * self.scale + self.loc


class Delta(distributions.Distribution):
    def __init__(self, dim, loc: float = 0.0, device="cpu") -> None:
        super().__init__()

        self.name = "delta"
        self.dim = dim
        self.loc = torch.tensor(loc, device=device, dtype=torch.float)
        self.scale = torch.tensor(0.0, device=device, dtype=torch.float)

    def sample(self, shape: tuple) -> torch.Tensor:
        return self.loc.repeat(*shape, self.dim)


class CenteredParticlesGauss(distributions.Distribution):
    """ Sample particles with zero center of mass
    """
    def __init__(
        self,
        n_particles,
        spatial_dim,
        scale: float = 1.0,
        device="cpu",
    ):
        super().__init__()
        self.n_particles = n_particles
        self.spatial_dim = spatial_dim
        self.dim = n_particles * spatial_dim
        # Centered distribution always has zero mean
        self.loc = torch.tensor(0.0, device=device, dtype=torch.float)
        self.scale = torch.tensor(scale, device=device, dtype=torch.float)
        self.device = device
        self.name = "meanfree"

    def sample(self, shape: tuple | None = None) -> torch.Tensor:
        if shape is None:
            shape = tuple()
        samples = torch.randn(*shape, self.dim, device=self.device) * self.scale
        samples = samples.reshape(-1, self.n_particles, self.spatial_dim)
        samples = samples - samples.mean(-2, keepdims=True)
        return samples.reshape(*shape, self.n_particles * self.spatial_dim)


class CenteredParticlesHarmonic(distributions.Distribution):
    """ Sample particles with zero center of mass from
        non-isotropic Gaussian based on a harmonic prior
        https://arxiv.org/pdf/2304.02198
    """
    def __init__(
        self,
        n_particles,
        spatial_dim,
        scale: float = 1.0,
        device="cpu",
    ):
        super().__init__()
        self.n_particles = n_particles
        self.spatial_dim = spatial_dim
        self.dim = n_particles * spatial_dim
        # Centered distribution always has zero mean
        self.loc = torch.tensor(0.0, device=device, dtype=torch.float)
        self.scale = torch.tensor(scale, device=device, dtype=torch.float)
        self.device = device
        self.name = "harmonic"

        cov = self._compute_cov(n_particles, spatial_dim)
        self.rank, self.A = self._decompose_svd(cov)

    def _compute_cov(self, n_particles, spatial_dim):
        """ e.g., n_particles = 2, spatial_dim = 3 would generate

            R = tensor([[  1,   0,   0, -0.5,   0,   0 ],
                        [  0,   1,   0,   0, -0.5,   0 ],
                        [  0,   0,   1,   0,    0, -0.5],
                        [-0.5,  0,   0,   1,    0,   0 ],
                        [  0, -0.5,   0,  0,    1,   0 ],
                        [  0,   0, -0.5,  0,    0,   1 ]])

            Denote x = [a1, a2, a3, b1, b2, b3]. This yields

            0.5 * x^T R x = a**2 + b**2 - ab = (a - b)**2
        """
        # TODO(ghliu) assume all particles are connected; otherwise changes A
        A = - 0.5 * torch.ones(n_particles, n_particles)
        A[torch.arange(n_particles), torch.arange(n_particles)] = 1.
        B = torch.eye(spatial_dim)
        return torch.kron(A, B).inverse()

    def _decompose_svd(self, cov):
        """ return the rank of cov and A where `cov = UΣV = AA^T`
        """
        U, S, Vt = torch.svd(cov)

        rank = (S > 1e-8).sum()
        A = U[:, :rank] @ torch.diag(S[:rank]).sqrt()
        return rank, A

    def sample(self, shape: tuple | None = None) -> torch.Tensor:
        """ generate samples by
            1. z ~ N(0,I) in the subspace with dim=rank
            2. x = Az, hence x ~ N(0, AA^T)
            3. make x zero COM
        """
        if shape is None:
            shape = tuple()

        B = math.prod(shape) # batch
        z = torch.randn(B, self.rank, device=self.device)
        samples = z @ self.A.to(z).T # (B, R) x (R, D) = (B, D)
        assert samples.shape == (B, self.dim)

        samples = samples * self.scale # note: scale = sqrt(alpha)

        samples = samples.reshape(*shape, self.n_particles, self.spatial_dim)
        samples = samples - samples.mean(-2, keepdims=True)
        return samples.reshape(*shape, self.dim)
