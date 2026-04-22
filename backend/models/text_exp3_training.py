"""
Text Experiment 2: Residual Transformer (Pre-LayerNorm + Residuals)
===================================================================
Architecture mirrors text_transformer.py with:
  ✦ Pre-LayerNorm (LN before attn and FFN)
  ✦ Residual connections (x = x + attn(x), x = x + ffn(x))
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

def stable_rank_weight(W: torch.Tensor) -> float:
    with torch.no_grad():
        try:
            W  = W.float().cpu()
            S  = torch.linalg.svdvals(W)
            sr = (S ** 2).sum() / (S[0] ** 2 + 1e-8)
            v  = sr.item()
            return v if np.isfinite(v) else 1.0
        except Exception:
            return 1.0


class SinusoidalPE(nn.Module):
    def __init__(self, d, max_len=512):
        super().__init__()
        pe  = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0,d,2).float() * (-math.log(10000.)/d))
        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:,:x.size(1)]

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x):
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / norm) * self.weight

class SwiGLUFeedForward(nn.Module):
    def __init__(self, d):
        super().__init__()
        hidden_dim = int(8 * d / 3)  # LLaMA style hidden dim
        self.w1 = nn.Linear(d, hidden_dim)
        self.w2 = nn.Linear(d, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, d)
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

def lambda_init_fn(layer_idx_1based):
    return 0.8 - 0.6 * math.exp(-0.3 * (layer_idx_1based - 1))

class DifferentialMultiHeadAttention(nn.Module):
    def __init__(self, d, h, layer_idx_1based, attn_dropout=0.1):
        super().__init__()
        self.num_heads = h
        self.head_dim = d // h
        
        self.q1 = nn.Linear(d, d)
        self.k1 = nn.Linear(d, d)
        self.q2 = nn.Linear(d, d)
        self.k2 = nn.Linear(d, d)
        self.v  = nn.Linear(d, d)
        self.proj = nn.Linear(d, d)

        self.lambda_q1 = nn.Parameter(torch.randn(self.head_dim) * 0.02)
        self.lambda_k1 = nn.Parameter(torch.randn(self.head_dim) * 0.02)
        self.lambda_q2 = nn.Parameter(torch.randn(self.head_dim) * 0.02)
        self.lambda_k2 = nn.Parameter(torch.randn(self.head_dim) * 0.02)

        self.lambda_init = lambda_init_fn(layer_idx_1based)
        self.head_norm = RMSNorm(self.head_dim)
        self.attn_drop = nn.Dropout(attn_dropout)

    def compute_lambda(self):
        lam1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1))
        lam2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2))
        return lam1 - lam2 + self.lambda_init

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q1 = self.q1(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k1 = self.k1(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q2 = self.q2(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k2 = self.k2(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v  = self.v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn1 = F.softmax((q1 @ k1.transpose(-2, -1)) * scale, dim=-1)
        attn2 = F.softmax((q2 @ k2.transpose(-2, -1)) * scale, dim=-1)

        # Apply dropout to each branch before subtraction
        attn1 = self.attn_drop(attn1)
        attn2 = self.attn_drop(attn2)

        lam = self.compute_lambda()
        diff_attn = attn1 - lam * attn2

        out = diff_attn @ v
        out = self.head_norm(out)
        out = out * (1.0 - self.lambda_init)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)

        return out

class DiffBlock(nn.Module):
    def __init__(self, d, h, layer_idx_1based):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.norm2 = RMSNorm(d)
        self.attn = DifferentialMultiHeadAttention(d, h, layer_idx_1based, attn_dropout=0.1)
        self.ffn = SwiGLUFeedForward(d)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x

class DiffTextTransformer(nn.Module):
    def __init__(self, vocab_size, d=128, h=4, L=6, num_cls=4, max_len=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d, padding_idx=0)
        self.pe = SinusoidalPE(d, max_len)
        self.blocks = nn.ModuleList([DiffBlock(d, h, layer_idx_1based=(i+1)) for i in range(L)])
        self.norm_f = RMSNorm(d)
        self.head = nn.Linear(d, num_cls)

    def forward(self, ids, mask=None):
        x = self.pe(self.embed(ids))
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.norm_f(x)
        return self.head(x[:,0])  # CLS token

    def compute_sr_per_layer(self) -> Dict[str, float]:
        """Stable rank per layer – averaged over q1, k1, q2, k2 projections."""
        sr_dict = {}
        for i, block in enumerate(self.blocks):
            q1_sr = stable_rank_weight(block.attn.q1.weight)
            k1_sr = stable_rank_weight(block.attn.k1.weight)
            q2_sr = stable_rank_weight(block.attn.q2.weight)
            k2_sr = stable_rank_weight(block.attn.k2.weight)
            sr_dict[f"layer_{i}"] = round((q1_sr + k1_sr + q2_sr + k2_sr) / 4.0, 4)
        return sr_dict

    def get_weights(self) -> Dict:
        return {n: p.detach().cpu().numpy() for n, p in self.named_parameters()}


# ── Training utilities ────────────────────────────────────────

def train_epoch(model: DiffTextTransformer,
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


def evaluate(model: DiffTextTransformer,
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
