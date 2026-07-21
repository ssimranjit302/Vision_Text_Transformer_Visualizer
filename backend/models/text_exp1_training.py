"""
Text Experiment 1: Vanilla Transformer (No Residual, No LayerNorm)
===================================================================
Architecture mirrors text_transformer.py but with:
  ✦ No residual connections  (x = attn(x) instead of x = x + attn(x))
  ✦ No LayerNorm             (no LN before attn or FFN)
  ✦ Standard scaled dot-product attention  (softmax)
  ✦ Sinusoidal positional encoding
  ✦ Stable rank computed on Q/K weight matrices every sr_interval steps

Dataset : AG News  → ./data/ag_news_train.csv  (train)
                   → ./data/ag_news_test.csv   (validation)
Classes : 4  (World=0, Sports=1, Business=2, Sci/Tech=3)
"""

import csv
import os
import re
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List

# ── Special tokens ────────────────────────────────────────────
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
CLS_TOKEN = "<CLS>"
SEP_TOKEN = "<SEP>"
PAD_IDX   = 0
UNK_IDX   = 1
CLS_IDX   = 2
SEP_IDX   = 3


# ── Vocabulary ────────────────────────────────────────────────

class SimpleVocabulary:
    """Mirror of text_transformer.py SimpleVocabulary."""
    def __init__(self, max_size: int = 30000):
        self.max_size = max_size
        self.word2idx: Dict[str, int] = {
            PAD_TOKEN: PAD_IDX,
            UNK_TOKEN: UNK_IDX,
            CLS_TOKEN: CLS_IDX,
            SEP_TOKEN: SEP_IDX,
        }
        self.idx2word: Dict[int, str] = {v: k for k, v in self.word2idx.items()}

    def build(self, texts: List[str]):
        counter: Counter = Counter()
        for text in texts:
            counter.update(self._tokenize(text))
        for word, _ in counter.most_common(self.max_size - 4):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx]  = word

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower().strip()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.split()

    def encode(self, text: str, max_len: int = 128):
        tokens = self._tokenize(text)
        token_ids = (
            [CLS_IDX]
            + [self.word2idx.get(t, UNK_IDX) for t in tokens][:max_len - 2]
            + [SEP_IDX]
        )
        token_ids = token_ids[:max_len]
        pad_len    = max_len - len(token_ids)
        token_ids += [PAD_IDX] * pad_len
        attention_mask = [1] * (max_len - pad_len) + [0] * pad_len
        return token_ids, attention_mask

    def __len__(self) -> int:
        return len(self.word2idx)


# ── Dataset ───────────────────────────────────────────────────

class TextDataset(Dataset):
    """Mirror of text_transformer.py TextDataset."""
    def __init__(self, texts: List[str], labels: List[int],
                 vocab: SimpleVocabulary, max_len: int = 128):
        self.texts  = texts
        self.labels = labels
        self.vocab  = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        token_ids, attention_mask = self.vocab.encode(self.texts[idx], self.max_len)
        return (
            torch.tensor(token_ids,      dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


# ── AG News loader ────────────────────────────────────────────

def _parse_ag_news(filepath: str):
    texts, labels = [], []
    with open(filepath, encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 3:
                try:
                    label = int(row[0]) - 1          # 1-4  →  0-3
                    text  = (row[1] + " " + row[2]).replace('\\', ' ').strip()
                    if 0 <= label <= 3 and text:
                        texts.append(text)
                        labels.append(label)
                except ValueError:
                    continue
    return texts, labels


def load_ag_news(data_dir: str = "./data"):
    """Load AG News from local CSV files in ./data/"""
    train_path = os.path.join(data_dir, "ag_news_train.csv")
    test_path  = os.path.join(data_dir, "ag_news_test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")

    train_texts, train_labels = _parse_ag_news(train_path)
    test_texts,  test_labels  = _parse_ag_news(test_path)

    print(f"[AG News] Train: {len(train_texts):,}  Test: {len(test_texts):,}")
    return train_texts, train_labels, test_texts, test_labels


def get_dataloader(batch_size: int = 128, max_len: int = 128):
    """Build vocab + dataloaders from the local CSV files."""
    train_texts, train_labels, test_texts, test_labels = load_ag_news()

    vocab = SimpleVocabulary(max_size=30000)
    vocab.build(train_texts)
    print(f"[Vocab] size={len(vocab):,}")

    train_ds = TextDataset(train_texts, train_labels, vocab, max_len)
    test_ds  = TextDataset(test_texts,  test_labels,  vocab, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)
    return train_loader, test_loader, vocab


# ── Stable Rank on weight matrices ────────────────────────────

def stable_rank_weight(W: torch.Tensor) -> float:
    """sr(W) = ‖W‖_F² / ‖W‖_2²  via SVD (on CPU)."""
    with torch.no_grad():
        try:
            W  = W.float().cpu()
            S  = torch.linalg.svdvals(W)
            sr = (S ** 2).sum() / (S[0] ** 2 + 1e-8)
            v  = sr.item()
            return v if np.isfinite(v) else 1.0
        except Exception:
            return 1.0


# ── Positional Encoding (sinusoidal) ─────────────────────────

class PositionalEncoding(nn.Module):
    """Identical to text_transformer.py PositionalEncoding."""
    def __init__(self, embed_dim: int, max_len: int = 512):
        super().__init__()
        pe       = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))   # (1, max_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


# ── Vanilla Multi-Head Attention (no dropout) ─────────────────
#   Based on MultiHeadSelfAttention from text_transformer.py
#   but dropout=0 and mask is accepted but never applied
#   (Vanilla: no masking, no extra regularisation)

class VanillaMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.q_proj  = nn.Linear(embed_dim, embed_dim)
        self.k_proj  = nn.Linear(embed_dim, embed_dim)
        self.v_proj  = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # Vanilla: no masking
        attn = torch.softmax(attn, dim=-1)
        out  = attn @ v
        out  = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


# ── Vanilla FFN (no dropout) ──────────────────────────────────
#   Matches text_transformer.py FeedForward but dropout=0

class VanillaFeedForward(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Vanilla Transformer Block  ────────────────────────────────
#   KEY: NO residual connections, NO LayerNorm
#   Compare with text_transformer.py TransformerBlock which does:
#       x = x + self.dropout(self.attn(self.ln1(x), mask))
#       x = x + self.dropout(self.ffn(self.ln2(x)))
#   Here we do only:
#       x = self.attn(x, mask)
#       x = self.ffn(x)

class VanillaTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attn = VanillaMultiHeadAttention(embed_dim, num_heads)
        self.ffn  = VanillaFeedForward(embed_dim)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        x = self.attn(x, mask)   # NO residual
        x = self.ffn(x)          # NO residual
        return x


# ── Vanilla Text Transformer ──────────────────────────────────

class VanillaTextTransformer(nn.Module):
    """
    Vanilla Transformer for text classification.
    Mirrors text_transformer.py TextTransformer but:
      - No LayerNorm  (ln1, ln2, ln_f removed)
      - No residuals  (x = f(x) not x = x + f(x))
      - No dropout    (dropout=0 everywhere)
    """
    def __init__(self, vocab_size: int, embed_dim: int = 128, num_heads: int = 4,
                 num_layers: int = 4, num_classes: int = 4, max_seq_len: int = 128):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.pos_embed   = PositionalEncoding(embed_dim, max_seq_len)
        self.blocks      = nn.ModuleList([
            VanillaTransformerBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        # NO ln_f (final LayerNorm)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids: torch.Tensor,
                attention_mask=None) -> torch.Tensor:
        x = self.token_embed(input_ids)   # (B, T, d)
        x = self.pos_embed(x)
        for block in self.blocks:
            x = block(x, attention_mask)
        cls = x[:, 0]                     # CLS token representation
        return self.head(cls)

    def compute_sr_per_layer(self) -> Dict[str, float]:
        """Stable rank per layer – averaged over Q and K projections."""
        sr_dict = {}
        for i, block in enumerate(self.blocks):
            q_sr = stable_rank_weight(block.attn.q_proj.weight)
            k_sr = stable_rank_weight(block.attn.k_proj.weight)
            sr_dict[f"layer_{i}"] = round((q_sr + k_sr) / 2.0, 4)
        return sr_dict

    def get_weights(self) -> Dict:
        return {n: p.detach().cpu().numpy() for n, p in self.named_parameters()}


# ── Training utilities ────────────────────────────────────────

def train_epoch(model: VanillaTextTransformer,
                train_loader: DataLoader,
                criterion,
                optimizer,
                device: str,
                compute_sr: bool = True,
                sr_interval: int  = 50,
                global_step: int  = 0) -> Dict:

    model.train()
    running_loss = 0.0
    correct = total = 0
    sr_per_layer: Dict[str, List[float]] = {}
    sr_overall:   List[float] = []
    sr_steps:     List[int]   = []

    for input_ids, attention_mask, targets in train_loader:
        input_ids, attention_mask, targets = (
            input_ids.to(device),
            attention_mask.to(device),
            targets.to(device),
        )
        global_step += 1

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss    = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item()
        _, predicted  = outputs.max(1)
        total   += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if compute_sr and (global_step % sr_interval == 0):
            sr_map  = model.compute_sr_per_layer()
            overall = sum(sr_map.values()) / len(sr_map)
            sr_steps.append(global_step)
            sr_overall.append(round(overall, 4))
            for k, v in sr_map.items():
                sr_per_layer.setdefault(k, []).append(v)

    return {
        "loss":        running_loss / len(train_loader),
        "accuracy":    100.0 * correct / total,
        "sr_per_layer": sr_per_layer,
        "sr_overall":   sr_overall,
        "sr_steps":     sr_steps,
        "global_step":  global_step,
    }


def evaluate(model: VanillaTextTransformer,
             test_loader: DataLoader,
             criterion,
             device: str) -> Dict:

    model.eval()
    running_loss = 0.0
    correct = total = 0

    with torch.no_grad():
        for input_ids, attention_mask, targets in test_loader:
            input_ids, attention_mask, targets = (
                input_ids.to(device),
                attention_mask.to(device),
                targets.to(device),
            )
            outputs = model(input_ids, attention_mask)
            loss    = criterion(outputs, targets)
            running_loss += loss.item()
            _, predicted  = outputs.max(1)
            total   += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return {
        "loss":     running_loss / len(test_loader),
        "accuracy": 100.0 * correct / total,
    }
