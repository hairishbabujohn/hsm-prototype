"""
Data loading for HSM: LRA ListOps (with HuggingFace fallback to synthetic datasets).
"""
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Synthetic datasets
# ---------------------------------------------------------------------------

def copy_memory_dataset(
    n_samples: int = 4000,
    seq_len: int = 64,
    vocab_size: int = 16,
    n_classes: int = 8,
    seed: int = 42,
) -> tuple:
    """Copy-memory task: recall a token at position 0 given a cue at end."""
    g = torch.Generator()
    g.manual_seed(seed)

    # Tokens 0..n_classes-1 are "content", n_classes..vocab_size-1 are fillers
    content_ids = torch.randint(0, n_classes, (n_samples,), generator=g)
    filler_id = n_classes
    seq = torch.full((n_samples, seq_len), filler_id, dtype=torch.long)
    seq[:, 0] = content_ids
    # Last token is a cue marker (vocab_size - 1)
    seq[:, -1] = vocab_size - 1
    labels = content_ids
    return seq, labels


def delayed_xor_dataset(
    n_samples: int = 4000,
    seq_len: int = 64,
    delay: int = 32,
    seed: int = 42,
) -> tuple:
    """XOR of tokens at position 0 and position `delay`."""
    g = torch.Generator()
    g.manual_seed(seed)

    seq = torch.randint(0, 2, (n_samples, seq_len), generator=g)
    labels = seq[:, 0] ^ seq[:, delay]  # XOR
    return seq.long(), labels.long()


def bracket_matching_dataset(
    n_samples: int = 4000,
    seq_len: int = 64,
    seed: int = 42,
) -> tuple:
    """Bracket matching: is the bracket sequence balanced? 50/50 split.
    0='(', 1=')'. Vectorized using Cycle Lemma for fast Dyck word generation."""
    import numpy as np
    rng = np.random.default_rng(seed)
    half = n_samples // 2
    half_n = seq_len // 2

    # ---- Balanced sequences via Cycle Lemma ----
    # For any permutation of half_n opens + half_n closes, rotating to start
    # right after the position of the minimum prefix sum yields a valid Dyck word.
    base = np.concatenate([np.zeros(half_n, dtype=np.int64),
                            np.ones(seq_len - half_n, dtype=np.int64)])
    balanced_seqs = np.tile(base, (half, 1))
    for i in range(half):
        rng.shuffle(balanced_seqs[i])
    delta = balanced_seqs * (-2) + 1
    cumsum = delta.cumsum(axis=1)
    # Find position of minimum prefix sum.
    # argmin gives the first (leftmost) occurrence; Cycle Lemma: rotate to start
    # right after this position to obtain a valid Dyck word.
    min_pos = np.argmin(cumsum, axis=1)  # (half,)
    # Rotate each sequence so it starts right after min_pos
    rotated = np.zeros_like(balanced_seqs)
    for i in range(half):
        rot = (min_pos[i] + 1) % seq_len
        rotated[i] = np.roll(balanced_seqs[i], -rot)

    # ---- Unbalanced sequences ----
    # Random sequences are almost always unbalanced
    unbalanced_seqs = rng.integers(0, 2, size=(half, seq_len))

    seqs = torch.tensor(np.concatenate([rotated, unbalanced_seqs], axis=0), dtype=torch.long)
    labels = torch.cat([torch.ones(half, dtype=torch.long), torch.zeros(half, dtype=torch.long)])

    perm = torch.randperm(n_samples, generator=torch.Generator().manual_seed(seed))
    return seqs[perm], labels[perm]


def _make_split(seqs: torch.Tensor, labels: torch.Tensor, train_frac: float = 0.8):
    n = len(seqs)
    split = int(n * train_frac)
    train_ds = TensorDataset(seqs[:split], labels[:split])
    val_ds = TensorDataset(seqs[split:], labels[split:])
    return train_ds, val_ds


def get_synthetic_dataloaders(
    dataset_name: str = "bracket",
    seq_len: int = 512,
    batch_size: int = 32,
    vocab_size: int = 32,
    n_classes: int = 2,
    seed: int = 42,
) -> tuple:
    if "bracket" in dataset_name:
        seqs, labels = bracket_matching_dataset(n_samples=4000, seq_len=seq_len, seed=seed)
        n_classes = 2
    elif "xor" in dataset_name:
        seqs, labels = delayed_xor_dataset(n_samples=4000, seq_len=seq_len, seed=seed)
        n_classes = 2
    else:
        seqs, labels = copy_memory_dataset(
            n_samples=4000, seq_len=seq_len, vocab_size=vocab_size,
            n_classes=n_classes, seed=seed
        )

    train_ds, val_ds = _make_split(seqs, labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, n_classes


# ---------------------------------------------------------------------------
# LRA ListOps via HuggingFace
# ---------------------------------------------------------------------------

class ListOpsDataset(Dataset):
    def __init__(self, hf_dataset, seq_len: int, vocab_size: int):
        self.data = hf_dataset
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = item["input_ids"]
        # Truncate / pad to seq_len
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        n = input_ids.shape[0]
        if n >= self.seq_len:
            input_ids = input_ids[:self.seq_len]
        else:
            pad = torch.zeros(self.seq_len - n, dtype=torch.long)
            input_ids = torch.cat([input_ids, pad])
        # Clamp to vocab range
        input_ids = input_ids.clamp(0, self.vocab_size - 1)
        label = torch.tensor(item["label"], dtype=torch.long)
        return input_ids, label


def get_lra_listops_dataloaders(
    seq_len: int = 2048,
    batch_size: int = 32,
    vocab_size: int = 32,
) -> tuple:
    from datasets import load_dataset
    ds = load_dataset("hf-internal-testing/long-range-arena", "listops")
    train_ds = ListOpsDataset(ds["train"], seq_len, vocab_size)
    val_ds = ListOpsDataset(ds["validation"], seq_len, vocab_size)
    n_classes = 10
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, n_classes


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def get_dataloaders(cfg: dict) -> tuple:
    """
    Returns (train_loader, val_loader, n_classes).
    cfg keys: dataset, seq_len, batch_size, vocab_size, n_classes, seed
    """
    dataset_name = cfg.get("dataset", "synthetic_bracket")
    seq_len = cfg.get("seq_len", 512)
    batch_size = cfg.get("batch_size", 32)
    vocab_size = cfg.get("vocab_size", 32)
    n_classes = cfg.get("n_classes", 2)
    seed = cfg.get("seed", 42)

    if dataset_name == "lra_listops":
        try:
            return get_lra_listops_dataloaders(
                seq_len=seq_len, batch_size=batch_size, vocab_size=vocab_size
            )
        except Exception as e:
            print(f"[data] LRA ListOps load failed ({e}), falling back to synthetic bracket.")
            dataset_name = "synthetic_bracket"
            n_classes = 2

    return get_synthetic_dataloaders(
        dataset_name=dataset_name,
        seq_len=seq_len,
        batch_size=batch_size,
        vocab_size=vocab_size,
        n_classes=n_classes,
        seed=seed,
    )
