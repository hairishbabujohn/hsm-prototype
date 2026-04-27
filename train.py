"""
Training script for the Homeostatic Sequence Model (HSM).
"""
import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from hsm.data import get_dataloaders
from hsm.losses import homeostatic_loss, gate_entropy
from hsm.model import HSMModel
from hsm.utils import Logger, amp_context, get_lr_scheduler, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train HSM")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "local_only", "global_only", "fixed_mix", "no_reg"],
        help="Ablation mode",
    )
    parser.add_argument("--run_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max_steps from config")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, n_classes: int) -> HSMModel:
    m = cfg["model"]
    return HSMModel(
        vocab_size=m["vocab_size"],
        d_model=m["d_model"],
        n_layers=m["n_layers"],
        n_classes=m.get("n_classes", n_classes),
        max_seq_len=m["max_seq_len"],
        ssm_rank=m.get("ssm_rank", 64),
        conv_kernels=m.get("conv_kernels", [3, 7, 15]),
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, force_mode: str) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x, force_mode=force_mode)
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    model.train()
    return correct / total if total > 0 else 0.0


def train(args):
    cfg = load_config(args.config)
    tcfg = cfg["training"]
    dcfg = cfg["data"]
    lcfg = cfg.get("loss", {})

    # Override max_steps if provided
    if args.max_steps is not None:
        tcfg["max_steps"] = args.max_steps

    seed = tcfg.get("seed", 42)
    set_seed(seed)

    # Setup run directory
    if args.run_dir is None:
        config_stem = Path(args.config).stem
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:22]  # microseconds to avoid collisions
        run_dir = os.path.join("runs", f"{config_stem}_{args.mode}_{ts}")
    else:
        run_dir = args.run_dir
    os.makedirs(run_dir, exist_ok=True)

    # Save config copy
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)

    logger = Logger(run_dir, run_name=os.path.basename(run_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}, mode={args.mode}, run_dir={run_dir}")

    # Data
    data_cfg = {
        **dcfg,
        "batch_size": tcfg["batch_size"],
        "vocab_size": cfg["model"]["vocab_size"],
        "n_classes": cfg["model"].get("n_classes", 2),
        "seed": seed,
    }
    train_loader, val_loader, n_classes = get_dataloaders(data_cfg)

    # Model
    model = build_model(cfg, n_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] params={n_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg.get("lr", 3e-4),
        betas=tuple(tcfg.get("betas", [0.9, 0.98])),
        weight_decay=tcfg.get("weight_decay", 0.01),
    )
    scheduler = get_lr_scheduler(
        optimizer,
        warmup_steps=tcfg.get("warmup_steps", 1000),
        max_steps=tcfg["max_steps"],
    )

    criterion = nn.CrossEntropyLoss()
    amp_enabled = tcfg.get("amp", False)
    grad_clip = tcfg.get("grad_clip", 1.0)
    log_every = tcfg.get("log_every", 100)
    eval_every = tcfg.get("eval_every", 1000)
    max_steps = tcfg["max_steps"]

    # Loss cfg
    loss_kwargs = {
        "tau": lcfg.get("tau", 0.6),
        "mu": lcfg.get("mu", 1.0),
        "lambda_": lcfg.get("lambda_", 1e-4),
        "g_min": lcfg.get("g_min", 0.05),
        "nu": lcfg.get("nu", 1.0),
        "mode": args.mode,
    }

    best_acc = 0.0
    step = 0
    train_iter = iter(train_loader)
    model.train()
    t0 = time.time()

    while step < max_steps:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        with amp_context(amp_enabled):
            logits, all_gates = model(x, force_mode=args.mode)
            task_loss = criterion(logits, y)
            total_loss, reg_loss, gate_metrics = homeostatic_loss(
                task_loss, all_gates, **loss_kwargs
            )

        total_loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        step += 1

        if step % log_every == 0:
            ent = gate_entropy(all_gates)
            metrics = {
                "task_loss": task_loss.item(),
                "reg_loss": reg_loss.item(),
                "total_loss": total_loss.item(),
                "gate_entropy": ent,
                "lr": scheduler.get_last_lr()[0],
                **gate_metrics,
            }
            elapsed = time.time() - t0
            print(
                f"step={step}/{max_steps} "
                f"loss={total_loss.item():.4f} "
                f"task={task_loss.item():.4f} "
                f"reg={metrics['reg_loss']:.4f} "
                f"util={gate_metrics.get('util', 0):.3f} "
                f"g_g={gate_metrics.get('mean_g_g', 0):.3f} "
                f"ent={ent:.3f} "
                f"elapsed={elapsed:.1f}s"
            )
            logger.log(metrics, step)

        if step % eval_every == 0:
            acc = evaluate(model, val_loader, device, force_mode=args.mode)
            print(f"  [eval] step={step} val_acc={acc:.4f}")
            logger.log({"val_acc": acc}, step)

            if acc > best_acc:
                best_acc = acc
                ckpt_path = os.path.join(run_dir, "best_model.pt")
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_acc": acc,
                        "config": cfg,
                        "mode": args.mode,
                    },
                    ckpt_path,
                )
                print(f"  [ckpt] saved best model (acc={acc:.4f}) -> {ckpt_path}")

    # Final eval
    final_acc = evaluate(model, val_loader, device, force_mode=args.mode)
    print(f"\n[done] final_val_acc={final_acc:.4f}, best_val_acc={best_acc:.4f}")
    logger.log({"final_acc": final_acc}, step)
    logger.close()

    # Save final summary
    with open(os.path.join(run_dir, "summary.txt"), "w") as f:
        f.write(f"mode={args.mode}\n")
        f.write(f"final_acc={final_acc:.4f}\n")
        f.write(f"best_acc={best_acc:.4f}\n")
        f.write(f"steps={step}\n")

    return final_acc


if __name__ == "__main__":
    args = parse_args()
    train(args)
