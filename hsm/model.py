"""
Homeostatic Sequence Model (HSM) - core model definition.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ISG(nn.Module):
    """Intrinsic State Gate: computes 4 token-geometry features and gates."""

    def __init__(self, d_model: int, momentum: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.d_model = d_model
        self.momentum = momentum
        self.eps = eps

        self.W_g = nn.Linear(4, 3, bias=True)

        # Running stats for z-scoring (not learnable params)
        self.register_buffer("running_mean", torch.zeros(4))
        self.register_buffer("running_var", torch.ones(4))

    def _compute_features(self, z: torch.Tensor) -> torch.Tensor:
        """Compute 4 token-geometry features. z: (B, N, d) -> phi: (B, N, 4)"""
        B, N, d = z.shape

        # 1. ||z_i||^2 - per token L2 squared
        f1 = (z ** 2).sum(dim=-1)  # (B, N)

        # 2. Var_seq(z) - sequence-wide variance scalar per sample, broadcast
        z_mean = z.mean(dim=1, keepdim=True)  # (B, 1, d)
        f2 = ((z - z_mean) ** 2).mean(dim=(1, 2)).unsqueeze(1).expand(B, N)  # (B, N)

        # 3. ||z_i - z_{i-1}||^2, z_{-1}=0
        z_prev = torch.cat([torch.zeros(B, 1, d, device=z.device, dtype=z.dtype), z[:, :-1, :]], dim=1)
        f3 = ((z - z_prev) ** 2).sum(dim=-1)  # (B, N)

        # 4. Var_ch(z_i) - variance across channels per token
        f4 = z.var(dim=-1, unbiased=False)  # (B, N)

        return torch.stack([f1, f2, f3, f4], dim=-1)  # (B, N, 4)

    def _update_running_stats(self, phi: torch.Tensor):
        """Update running mean/var using EMA. phi: (B, N, 4)"""
        flat = phi.detach().reshape(-1, 4)
        batch_mean = flat.mean(dim=0)
        # Use Bessel-uncorrected variance for consistency with running stats
        batch_var = flat.var(dim=0, unbiased=False)
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

    def forward(self, z: torch.Tensor, force_mode: str = "full"):
        """z: (B, N, d) -> gates (B, N, 3), phi_norm (B, N, 4)"""
        phi = self._compute_features(z)  # (B, N, 4)

        if self.training:
            self._update_running_stats(phi)

        # Z-score normalize features
        std = (self.running_var + self.eps).sqrt()
        phi_norm = (phi - self.running_mean) / std  # (B, N, 4)

        if force_mode == "local_only":
            B, N, _ = z.shape
            gates = torch.zeros(B, N, 3, device=z.device, dtype=z.dtype)
            gates[..., 0] = 1.0
        elif force_mode == "global_only":
            B, N, _ = z.shape
            gates = torch.zeros(B, N, 3, device=z.device, dtype=z.dtype)
            gates[..., 1] = 1.0
        elif force_mode == "fixed_mix":
            B, N, _ = z.shape
            gates = torch.zeros(B, N, 3, device=z.device, dtype=z.dtype)
            gates[..., 0] = 0.5
            gates[..., 1] = 0.5
        else:
            logits = self.W_g(phi_norm)  # (B, N, 3)
            gates = F.softmax(logits, dim=-1)

        return gates, phi_norm


class LocalMixer(nn.Module):
    """Multi-scale depthwise separable convolutions along sequence dimension."""

    def __init__(self, d_model: int, kernel_sizes: list = None):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 7, 15]
        self.kernel_sizes = kernel_sizes

        # For each kernel: depthwise conv + pointwise conv
        self.depthwise_convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=k, padding=k // 2, groups=d_model)
            for k in kernel_sizes
        ])
        self.pointwise_convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=1)
            for _ in kernel_sizes
        ])
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, N, d) -> (B, N, d)"""
        # Conv1d expects (B, d, N)
        x = z.transpose(1, 2)  # (B, d, N)
        N = x.shape[-1]

        out = None
        for dw, pw in zip(self.depthwise_convs, self.pointwise_convs):
            h = dw(x)
            # Trim to exact length in case of any off-by-one
            h = h[..., :N]
            h = pw(h)
            out = h if out is None else out + h

        out = out.transpose(1, 2)  # (B, N, d)
        return self.output_proj(out)


class GlobalMixer(nn.Module):
    """Diagonal SSM (state-space model) global mixer."""

    def __init__(self, d_model: int, ssm_rank: int = 64):
        super().__init__()
        self.d_model = d_model
        self.ssm_rank = ssm_rank

        self.a_log = nn.Parameter(torch.zeros(ssm_rank))
        self.dt_log = nn.Parameter(torch.full((d_model,), -1.0))
        self.B = nn.Parameter(torch.randn(d_model, ssm_rank) * 0.01)
        self.C = nn.Parameter(torch.randn(ssm_rank, d_model) * 0.01)
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, N, d) -> (B, N, d)"""
        B_size, N, d = z.shape
        r = self.ssm_rank

        A = -F.softplus(self.a_log)          # (r,)
        dt = F.softplus(self.dt_log)          # (d,)

        # Abar[j, k] = exp(A[k] * dt[j]), shape (d, r)
        Abar = torch.exp(A.unsqueeze(0) * dt.unsqueeze(1))  # (d, r)

        # Precompute u[b, t, j, k] = z[b, t, j] * B[j, k] -> (B, N, d, r)
        u = z.unsqueeze(-1) * self.B.unsqueeze(0).unsqueeze(0)  # (B, N, d, r)

        # C transposed for output: (d, r)
        C_t = self.C.t()
        Abar_b = Abar.unsqueeze(0)  # (1, d, r)

        h = torch.zeros(B_size, d, r, device=z.device, dtype=z.dtype)
        outputs = []
        for t in range(N):
            h = Abar_b * h + u[:, t]                              # (B, d, r)
            y_t = (h * C_t.unsqueeze(0)).sum(dim=-1) + z[:, t] * self.D  # (B, d)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # (B, N, d)


class HSMLayer(nn.Module):
    """One layer of the Homeostatic Sequence Model."""

    def __init__(self, d_model: int, ssm_rank: int = 64, conv_kernels: list = None):
        super().__init__()
        if conv_kernels is None:
            conv_kernels = [3, 7, 15]

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.isg = ISG(d_model)
        self.local_mixer = LocalMixer(d_model, conv_kernels)
        self.global_mixer = GlobalMixer(d_model, ssm_rank)
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, force_mode: str = "full"):
        """x: (B, N, d) -> (B, N, d), gates (B, N, 3)"""
        z = self.norm1(x)

        gates, _ = self.isg(z, force_mode=force_mode)
        g_l = gates[..., 0:1]  # (B, N, 1)
        g_g = gates[..., 1:2]  # (B, N, 1)

        local_out = self.local_mixer(z)
        global_out = self.global_mixer(z)

        y = g_l * local_out + g_g * global_out
        x_next = x + self.alpha * self.norm2(y)

        return x_next, gates


class HSMModel(nn.Module):
    """Full Homeostatic Sequence Model."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_classes: int,
        max_seq_len: int = 2048,
        ssm_rank: int = 64,
        conv_kernels: list = None,
    ):
        super().__init__()
        if conv_kernels is None:
            conv_kernels = [3, 7, 15]

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList([
            HSMLayer(d_model, ssm_rank, conv_kernels)
            for _ in range(n_layers)
        ])

        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, input_ids: torch.Tensor, force_mode: str = "full"):
        """
        input_ids: (B, N) int tensor
        Returns: logits (B, n_classes), list of per-layer gates (B, N, 3)
        """
        B, N = input_ids.shape
        pos = torch.arange(N, device=input_ids.device).unsqueeze(0)  # (1, N)

        x = self.token_emb(input_ids) + self.pos_emb(pos)  # (B, N, d)

        all_gates = []
        for layer in self.layers:
            x, gates = layer(x, force_mode=force_mode)
            all_gates.append(gates)

        x = self.norm_out(x)
        # Pool over sequence dimension for classification
        x = x.mean(dim=1)  # (B, d)
        logits = self.head(x)  # (B, n_classes)

        return logits, all_gates
