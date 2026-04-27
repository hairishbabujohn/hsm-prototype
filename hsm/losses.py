"""
Homeostatic loss: task loss + regularization over gate distributions.
"""
import torch
import torch.nn.functional as F


def homeostatic_loss(
    task_loss: torch.Tensor,
    all_gates: list,
    tau: float = 0.6,
    mu: float = 1.0,
    lambda_: float = 1e-4,
    g_min: float = 0.05,
    nu: float = 1.0,
    mode: str = "full",
) -> tuple:
    """
    Compute total loss = task_loss + reg.

    Args:
        task_loss: scalar task loss
        all_gates: list of (B, N, 3) gate tensors, one per layer
        tau: target utilization
        mu: penalty weight for deviation from tau
        lambda_: sparsity weight on global gate
        g_min: minimum global gate utilization floor
        nu: weight for floor penalty
        mode: if 'no_reg', skip regularization

    Returns:
        total_loss, reg_loss, metrics dict
    """
    zero = torch.zeros(1, device=task_loss.device)

    if len(all_gates) == 0:
        return task_loss, zero, {}

    # Aggregate gates across layers - always compute metrics
    g_l_list, g_g_list, g_s_list = [], [], []
    for gates in all_gates:
        g_l_list.append(gates[..., 0])
        g_g_list.append(gates[..., 1])
        g_s_list.append(gates[..., 2])

    g_l = torch.stack(g_l_list, dim=0).mean()
    g_g = torch.stack(g_g_list, dim=0).mean()
    g_s = torch.stack(g_s_list, dim=0).mean()
    util = (torch.stack(g_l_list, dim=0) + torch.stack(g_g_list, dim=0)).mean()

    metrics = {
        "util": util.item(),
        "mean_g_l": g_l.item(),
        "mean_g_g": g_g.item(),
        "mean_g_s": g_s.item(),
    }

    if mode == "no_reg":
        return task_loss, zero, metrics

    reg = (
        lambda_ * g_g
        + mu * (util - tau) ** 2
        + nu * F.relu(g_min - g_g) ** 2
    )

    total_loss = task_loss + reg
    return total_loss, reg, metrics


def gate_entropy(all_gates: list) -> float:
    """Compute average gate entropy across all layers, B, N."""
    entropies = []
    for gates in all_gates:
        # gates: (B, N, 3)
        H = -(gates * torch.log(gates + 1e-8)).sum(dim=-1)  # (B, N)
        entropies.append(H.mean().item())
    return sum(entropies) / len(entropies) if entropies else 0.0
