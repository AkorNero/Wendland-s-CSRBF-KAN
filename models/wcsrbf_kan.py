import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def wendland_phi_1d(r: torch.Tensor, k: int) -> torch.Tensor:
    # r >= 0
    t = (1.0 - r).clamp(min=0.0)
    if k == 0:
        return t
    elif k == 1:
        return (t ** 3) * (1.0 + 3.0*r)
    elif k == 2:
        return (t ** 5) * (1.0 + 5.0*r + 8.0*r**2)
    elif k == 3:
        return (t ** 7) * (1.0 + 7.0*r + 19.0*r**2 + 21.0*r**3)
    elif k == 4:
        return (t ** 9) * (1.0 + 9.0*r + 33.86*r**2 + 59.71*r**3 + 54.86*r**4)
    else:
        raise ValueError("Supported smoothness k ∈ {0,1,2,3,4} for d=1")

class WendlandCSRBF(nn.Module):
    """
    Per-feature CSRBF features:
      input  x: (B, in_features)
      output phi: (B, in_features * n_centers)

    Options:
      - trainable_centers (bool)
      - trainable_sigma   (bool)
      - sigma is a direct positive parameter (no softplus); clamped in sigma()
      - centers initialized uniformly in `center_range`
      - sigma initialized from uniform grid spacing (or `init_sigma` if M=1)

    Plus:
      - reset_centers_sigma_from_data(x, grid_eps, ...) mixes uniform and adaptive (quantile) grids from data.
    """
    def __init__(
        self,
        in_features,
        n_centers=8,
        k=2,
        center_range=(-2.0, 2.0),
        per_feature_centers=True,
        trainable_centers=True,
        trainable_sigma=True,
        init_sigma=1.0,     # used when n_centers == 1
        min_sigma=1e-3,
        s_scale=1.0,        # sigma ≈ s_scale * (uniform grid spacing)
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.n_centers = int(n_centers)
        self.k = int(k)
        self.min_sigma = float(min_sigma)
        self.s_scale = float(s_scale)
        self._center_range = tuple(center_range)
        self._per_feature_centers = bool(per_feature_centers)

        # ---- uniform centers in center_range ----
        lo, hi = self._center_range
        uni = torch.linspace(lo, hi, self.n_centers).unsqueeze(0)  # (1, M)
        if self._per_feature_centers:
            uni = uni.repeat(self.in_features, 1)                  # (D, M)

        # centers parameter (trainable or frozen)
        self.centers = nn.Parameter(uni, requires_grad=bool(trainable_centers))

        # ---- sigma parameter (direct positive param; no softplus) ----
        if self.n_centers > 1:
            # uniform grid spacing
            step = (hi - lo) / (self.n_centers - 1)
            base_sigma = max(self.min_sigma, self.s_scale * float(step))
            sigma0 = torch.full((self.in_features, self.n_centers), base_sigma)
        else:
            sigma0 = torch.full((self.in_features, self.n_centers), float(max(self.min_sigma, init_sigma)))

        self.sigma_param = nn.Parameter(sigma0, requires_grad=bool(trainable_sigma))

    @torch.no_grad()
    def reset_centers_sigma_from_data(
        self,
        x,                     # (N, D) data already in THIS block's space (apply LN outside if you use LN)
        grid_eps=0.5,          # mix weight: eps*uniform + (1-eps)*adaptive
        low_q=0.01,
        high_q=0.99,
        s_scale=None          # optional override for sigma scaling based on spacing
    ):
        """
        Reinitialize centers & sigma from data using a mix:
          centers = grid_eps * uniform + (1-grid_eps) * adaptive_quantiles
        Sigma per feature = (s_scale or self.s_scale) * median neighbor spacing (clamped by min_sigma).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        x_cpu = x.detach().cpu()

        D, M = self.in_features, self.n_centers
        centers_uni = torch.empty(D, M)
        centers_adp = torch.empty(D, M)

        for i in range(D):
            xi = x_cpu[:, i]
            lo = torch.quantile(xi, low_q)
            hi = torch.quantile(xi, high_q)
            centers_uni[i] = torch.linspace(lo.item(), hi.item(), M)
            # adaptive = equal-mass quantiles (exclude 0 and 1)
            qs = torch.linspace(0.0, 1.0, M+2)[1:-1]
            centers_adp[i] = torch.quantile(xi, qs)

        centers_mix = float(grid_eps) * centers_uni + (1.0 - float(grid_eps)) * centers_adp

        # sigma from spacing
        scale = self.s_scale if s_scale is None else float(s_scale)
        if M > 1:
            diffs = centers_mix[:, 1:] - centers_mix[:, :-1]
            med = diffs.median(dim=1, keepdim=True).values.clamp_min(1e-8)
            sigma = (scale * med).expand(D, M)
            sigma[:, -1] = sigma[:, -1]
        else:
            sigma = torch.full((D, M), max(self.min_sigma, scale))

        sigma = sigma.clamp_min(self.min_sigma)

        # match parameter shapes/dev/dtype
        dev, dt = self.centers.device, self.centers.dtype
        if self.centers.shape[0] == 1:  # per_feature_centers=False --> broadcasted centers
            centers_mix = centers_mix.mean(dim=0, keepdim=True)
        self.centers.data = centers_mix.to(device=dev, dtype=dt)
        self.sigma_param.data = sigma.to(device=dev, dtype=dt)

    def sigma(self):
        # ensure strictly positive scales
        return self.sigma_param.clamp_min(self.min_sigma)

    def forward(self, x):
        # x: (B, D)
        B = x.shape[0]
        x_exp = x.unsqueeze(-1)                          # (B, D, 1)
        c = self.centers.unsqueeze(0)                    # (1, D or 1, M) -> broadcasts to (B, D, M)
        s = self.sigma().unsqueeze(0)                    # (1, D, M)
        r = (x_exp - c).abs() / s                        # (B, D, M)
        phi = wendland_phi_1d(r, self.k)                 # (B, D, M)
        return phi.reshape(B, self.in_features * self.n_centers)
    
class WCSRBFKANLayer(nn.Module):
    """
    y = (Linear(SiLU(norm(x))) ⊙ Wb_post) + (Linear(phi(norm(x))) ⊙ Ws_post)

    Options:
      - enable_layer_norm (bool)
      - post_mix_mode: "per_out" (vector) or "scalar" (global)
      - CSRBF block with trainable/frozen centers & sigma
      - data-based grid reset helper (handles LN inside)
    """
    def __init__(
        self,
        in_features,
        out_features,
        n_centers=8,
        k=2,
        enable_layer_norm=True,
        post_mix_mode="per_out",
        center_range=(-2.0, 2.0),
        per_feature_centers=True,
        trainable_centers=True,
        trainable_sigma=True,
        init_sigma=1.0,
        min_sigma=1e-3,
        s_scale=1.0,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.post_mix_mode = post_mix_mode
        self.enable_layer_norm = bool(enable_layer_norm)

        self.layernorm = nn.LayerNorm(self.in_features) if self.enable_layer_norm else nn.Identity()

        # base (SiLU) linear
        self.base_weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        # nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.xavier_uniform_(self.base_weight)

        # CSRBF features + linear
        self.csrbf = WendlandCSRBF(
            self.in_features, n_centers=n_centers, k=k,
            center_range=center_range,
            per_feature_centers=per_feature_centers,
            trainable_centers=trainable_centers,
            trainable_sigma=trainable_sigma,
            init_sigma=init_sigma,
            min_sigma=min_sigma,
            s_scale=s_scale,
        )
        self.spline_weight = nn.Parameter(torch.empty(self.out_features, self.in_features * n_centers))
        nn.init.xavier_uniform_(self.spline_weight)

        # post-mix weights
        if post_mix_mode == "per_out":
            self.Wb_post = nn.Parameter(torch.ones(self.out_features))
            self.Ws_post = nn.Parameter(torch.ones(self.out_features))
        elif post_mix_mode == "scalar":
            self.Wb_post = nn.Parameter(torch.tensor(1.0))
            self.Ws_post = nn.Parameter(torch.tensor(1.0))
        else:
            raise ValueError("post_mix_mode must be 'per_out' or 'scalar'")

    @torch.no_grad()
    def init_wcsrbf_grid(self, x_sample, grid_eps=0.5, low_q=0.01, high_q=0.99, s_scale=None):
        """
        Convenience wrapper: reinitialize centers & sigma from a data minibatch.
        If LayerNorm is enabled, apply it before passing to CSRBF reset.
        """
        z = self.layernorm(x_sample) if self.enable_layer_norm else x_sample
        self.csrbf.reset_centers_sigma_from_data(
            z, grid_eps=float(grid_eps), low_q=low_q, high_q=high_q, s_scale=s_scale
        )

    def forward(self, x):
        z = self.layernorm(x)
        base = F.linear(F.silu(z), self.base_weight)           # (B, O)
        phi  = self.csrbf(z)                                   # (B, D*M)
        spline = F.linear(phi, self.spline_weight)             # (B, O)

        if self.post_mix_mode == "per_out":
            y = base * self.Wb_post.view(1, -1) + spline * self.Ws_post.view(1, -1)
        else:
            y = base * self.Wb_post + spline * self.Ws_post
        return y

class WCSRBFKAN(nn.Module):
    """
    dims: [D_in, H1, ..., D_out]
    """
    def __init__(
        self,
        dims,
        n_centers=8,
        k=2,
        enable_layer_norm=True,
        trainable_centers=True,
        trainable_sigma=True,
        post_mix_mode="per_out",
        center_range=(-2.0, 2.0),
        per_feature_centers=True,
        init_sigma=1.0,
        min_sigma=1e-3,
        s_scale=1.0,
        grid_eps = 1.0
    ):
        super().__init__()
        self.dims = list(dims)
        L = len(self.dims) - 1

        assert grid_eps >= 0 and grid_eps <= 1

        self.grid_eps = grid_eps

        def to_list(v):
            return v if isinstance(v, (list, tuple)) else [v] * L

        n_centers_list = to_list(n_centers)
        k_list = to_list(k)
        ln_list = to_list(enable_layer_norm)
        pm_list = to_list(post_mix_mode)
        self.trainable_sigma = trainable_sigma
        self.trainable_centers = trainable_centers

        layers = []
        for i in range(L):
            in_dim, out_dim = self.dims[i], self.dims[i+1]
            layers.append(
                WCSRBFKANLayer(
                    in_features=in_dim,
                    out_features=out_dim,
                    n_centers=int(n_centers_list[i]),
                    k=int(k_list[i]),
                    enable_layer_norm=bool(ln_list[i]),
                    post_mix_mode=str(pm_list[i]),
                    center_range=center_range,
                    per_feature_centers=per_feature_centers,
                    trainable_centers=trainable_centers,
                    trainable_sigma=trainable_sigma,
                    init_sigma=init_sigma,
                    min_sigma=min_sigma,
                    s_scale=s_scale,
                )
            )
        self.layers = nn.ModuleList(layers)

    def sigma_inverse_l2(self, lambda_sigma=1e-3, eps=1e-8, squared=True):
        """
        L2 penalty on the inverse of sigma.
        squared=True  ->  sum_i (1 / sigma_i)^2       (squared L2, typical)
        squared=False ->  || 1 / sigma ||_2           (true L2 norm)

        Uses a smooth, positive surrogate for sigma so gradients don't vanish
        at the clamp floor used in forward().
        """
        reg = 0.0
        for layer in self.layers:
            s_raw = layer.csrbf.sigma_param
            s_min = layer.csrbf.min_sigma
            # smooth lower bound: > s_min and differentiable
            s_safe = F.softplus(s_raw - s_min) + s_min
            inv = 1.0 / (s_safe + eps)

            if squared:
                reg = reg + (inv * inv).sum()          # squared L2 on inverse
            else:
                reg = reg + torch.linalg.norm(inv, 2)  # true L2 norm
        return lambda_sigma * reg
    
    def centers_l2(self, lambda_c=1e-5):
        return lambda_c * sum((l.csrbf.centers.pow(2).sum() for l in self.layers))

    def weights_l2(model, lambda_w=1e-4):
        """
        L2 on base/spline weights + (optionally) post-mix scalars/vectors.
        lambda_w     : coeff for base_weight & spline_weight
        """

        reg_w = 0.0
        reg_post = 0.0
        for l in model.layers:
            # linear weights
            reg_w += l.base_weight.pow(2).sum() + l.spline_weight.pow(2).sum()
            # post-mix (exists for both 'per_out' and 'scalar' modes)
            reg_post += l.Wb_post.pow(2).sum() + l.Ws_post.pow(2).sum()

        return lambda_w * (reg_w + reg_post)



    @property
    def n_layers(self):
        return len(self.layers)
    
    @property
    def trainable_centers_bool(self):
        return bool(self.trainable_centers)
    
    @property
    def trainable_sigma_bool(self):
        return bool(self.trainable_sigma)

    def layer(self, idx):
        return self.layers[idx]

    def forward(self, x):
        for layer in self.layers:
            if self.grid_eps != 1:
                layer.init_wcsrbf_grid(x, grid_eps = self.grid_eps)
            x = layer(x)
        return x
