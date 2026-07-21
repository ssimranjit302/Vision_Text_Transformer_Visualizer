# ═══════════════════════════════════════════════════════════════════════════════
# TEXT EXPERIMENT 2 — Residual Transformer (Pre-LayerNorm + Residuals)
# Dataset  : AG News  |  GPU: T4  |  30 Epochs
# Params   : n_embd=128, n_head=4, n_layer=4, LR=3e-4
# ═══════════════════════════════════════════════════════════════════════════════

import csv, os, re, math, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ── Confirm GPU ──────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
N_EMBD    = 128
N_HEAD    = 4
N_LAYER   = 4
EPOCHS    = 30
LR        = 3e-4
BATCH     = 256        # larger batch for T4 efficiency
MAX_LEN   = 128
VOCAB_SZ  = 30000
NUM_CLS   = 4
SR_INTERVAL = 50       # compute stable rank every N steps

# ══════════════════════════════════════════════════════════════════════════════
# VOCABULARY
# ══════════════════════════════════════════════════════════════════════════════
PAD_IDX, UNK_IDX, CLS_IDX, SEP_IDX = 0, 1, 2, 3

class SimpleVocabulary:
    def __init__(self, max_size=30000):
        self.max_size = max_size
        self.word2idx = {"<PAD>":0,"<UNK>":1,"<CLS>":2,"<SEP>":3}
        self.idx2word = {v:k for k,v in self.word2idx.items()}

    def build(self, texts):
        counter = Counter()
        for t in texts: counter.update(self._tok(t))
        for w,_ in counter.most_common(self.max_size - 4):
            idx = len(self.word2idx)
            self.word2idx[w] = idx
            self.idx2word[idx] = w

    def _tok(self, text):
        return re.sub(r'[^a-z0-9\s]','',text.lower().strip()).split()

    def encode(self, text, max_len=128):
        ids = [CLS_IDX]+[self.word2idx.get(t,UNK_IDX) for t in self._tok(text)][:max_len-2]+[SEP_IDX]
        ids = ids[:max_len]
        pad = max_len - len(ids)
        return ids+[PAD_IDX]*pad, [1]*(max_len-pad)+[0]*pad

    def __len__(self): return len(self.word2idx)

# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════
class AGNewsDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=128):
        self.texts, self.labels, self.vocab, self.max_len = texts, labels, vocab, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        ids, mask = self.vocab.encode(self.texts[idx], self.max_len)
        return (torch.tensor(ids,dtype=torch.long),
                torch.tensor(mask,dtype=torch.long),
                torch.tensor(self.labels[idx],dtype=torch.long))

def parse_ag_news(path):
    texts, labels = [], []
    with open(path, encoding='utf-8', errors='replace') as f:
        for row in csv.reader(f):
            if len(row) >= 3:
                try:
                    lbl = int(row[0]) - 1
                    txt = (row[1]+" "+row[2]).replace('\\',' ').strip()
                    if 0<=lbl<=3 and txt:
                        texts.append(txt); labels.append(lbl)
                except: pass
    return texts, labels

# ── Kaggle AG News paths ──────────────────────────────────────────────────────
# Upload ag_news_train.csv and ag_news_test.csv to Kaggle input
TRAIN_PATH = "/kaggle/input/datasets/ssimranjit302/agnewsdataset/ag_news_train.csv"
TEST_PATH  = "/kaggle/input/datasets/ssimranjit302/agnewsdataset/ag_news_test.csv"

# If using custom upload, change paths:
# TRAIN_PATH = "/kaggle/input/your-dataset/ag_news_train.csv"
# TEST_PATH  = "/kaggle/input/your-dataset/ag_news_test.csv"

print("Loading AG News...")
train_texts, train_labels = parse_ag_news(TRAIN_PATH)
test_texts,  test_labels  = parse_ag_news(TEST_PATH)
print(f"Train: {len(train_texts):,}  |  Test: {len(test_texts):,}")

vocab = SimpleVocabulary(VOCAB_SZ)
vocab.build(train_texts)
print(f"Vocab size: {len(vocab):,}")

train_ds = AGNewsDataset(train_texts, train_labels, vocab, MAX_LEN)
test_ds  = AGNewsDataset(test_texts,  test_labels,  vocab, MAX_LEN)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)
print(f"Train batches: {len(train_loader)}  |  Test batches: {len(test_loader)}")

# ══════════════════════════════════════════════════════════════════════════════
# STABLE RANK
# ══════════════════════════════════════════════════════════════════════════════
def stable_rank_weight(W):
    with torch.no_grad():
        try:
            W_ = W.float().cpu()
            S  = torch.linalg.svdvals(W_)
            sr = (S**2).sum() / (S[0]**2 + 1e-8)
            v  = sr.item()
            return v if np.isfinite(v) else 1.0
        except: return 1.0

# ══════════════════════════════════════════════════════════════════════════════
# MODEL — Residual Transformer (LayerNorm + Residuals)
# ══════════════════════════════════════════════════════════════════════════════
class SinusoidalPE(nn.Module):
    def __init__(self, d, max_len=512):
        super().__init__()
        pe  = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0,d,2).float() * (-math.log(10000.)/d))
        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:,:x.size(1)]

class ResidualMHA(nn.Module):
    """Multi-head attention — standard."""
    def __init__(self, d, h):
        super().__init__()
        self.h, self.dh = h, d//h
        self.scale = (d//h)**-0.5
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
    def forward(self, x, mask=None):
        B,T,C = x.shape
        q = self.q(x).view(B,T,self.h,self.dh).transpose(1,2)
        k = self.k(x).view(B,T,self.h,self.dh).transpose(1,2)
        v = self.v(x).view(B,T,self.h,self.dh).transpose(1,2)
        a = torch.softmax((q @ k.transpose(-2,-1))*self.scale, dim=-1)
        o = (a @ v).transpose(1,2).contiguous().view(B,T,C)
        return self.o(o)

class ResidualFFN(nn.Module):
    """Feed-forward — standard."""
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,4*d), nn.GELU(), nn.Linear(4*d,d))
    def forward(self, x): return self.net(x)

class ResidualBlock(nn.Module):
    """Residual connections + Pre-LayerNorm."""
    def __init__(self, d, h):
        super().__init__()
        self.ln1  = nn.LayerNorm(d)
        self.attn = ResidualMHA(d, h)
        self.ln2  = nn.LayerNorm(d)
        self.ffn  = ResidualFFN(d)
    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)  # Residual + Pre-LN
        x = x + self.ffn(self.ln2(x))         # Residual + Pre-LN
        return x

class ResidualTextTransformer(nn.Module):
    def __init__(self, vocab_size, d=128, h=4, L=4, num_cls=4, max_len=128):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, d, padding_idx=PAD_IDX)
        self.pe     = SinusoidalPE(d, max_len)
        self.blocks = nn.ModuleList([ResidualBlock(d,h) for _ in range(L)])
        self.ln_f   = nn.LayerNorm(d)         # Final LN
        self.head   = nn.Linear(d, num_cls) 

    def forward(self, ids, mask=None):
        x = self.pe(self.embed(ids))
        for blk in self.blocks: x = blk(x, mask)
        x = self.ln_f(x)
        return self.head(x[:,0])  # CLS token

    def sr_per_layer(self):
        sr = {}
        for i, blk in enumerate(self.blocks):
            qsr = stable_rank_weight(blk.attn.q.weight)
            ksr = stable_rank_weight(blk.attn.k.weight)
            sr[f"layer_{i}"] = round((qsr+ksr)/2, 4)
        return sr

# Build model
model = ResidualTextTransformer(len(vocab), d=N_EMBD, h=N_HEAD, L=N_LAYER,
                                num_cls=NUM_CLS, max_len=MAX_LEN).to(device)
print(f"\nModel params: {sum(p.numel() for p in model.parameters()):,}")

# Verify LayerNorm count
ln_count = sum(1 for _,m in model.named_modules() if isinstance(m, nn.LayerNorm))
print(f"LayerNorm count: {ln_count}  ← Should be {N_LAYER * 2 + 1}")

# SR at init
sr_init = model.sr_per_layer()
print(f"SR at init: {sr_init}")

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

# History accumulator
history = {
    "train_loss": [], "train_acc": [],
    "val_loss":   [], "val_acc":   [],
    "sr_per_layer": {}, "sr_overall": [], "sr_steps": [],
    "total_epochs": EPOCHS
}
global_step = 0

def train_epoch(ep):
    global global_step
    model.train()
    loss_sum = correct = total = 0
    ep_sr_layer = {}; ep_sr_overall = []; ep_sr_steps = []

    for ids, mask, tgt in train_loader:
        ids, mask, tgt = ids.to(device), mask.to(device), tgt.to(device)
        global_step += 1
        optimizer.zero_grad()
        out  = model(ids, mask)
        loss = criterion(out, tgt)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_sum += loss.item()
        pred = out.argmax(1)
        total   += tgt.size(0)
        correct += pred.eq(tgt).sum().item()

        if global_step % SR_INTERVAL == 0:
            sr_map = model.sr_per_layer()
            ep_sr_steps.append(global_step)
            ep_sr_overall.append(round(sum(sr_map.values())/len(sr_map),4))
            for k,v in sr_map.items():
                ep_sr_layer.setdefault(k,[]).append(v)

    return {
        "loss": round(loss_sum/len(train_loader),4),
        "acc":  round(100.*correct/total,2),
        "sr_layer": ep_sr_layer,
        "sr_overall": ep_sr_overall,
        "sr_steps": ep_sr_steps,
    }

def evaluate():
    model.eval()
    loss_sum = correct = total = 0
    with torch.no_grad():
        for ids, mask, tgt in test_loader:
            ids, mask, tgt = ids.to(device), mask.to(device), tgt.to(device)
            out  = model(ids, mask)
            loss = criterion(out, tgt)
            loss_sum += loss.item()
            pred = out.argmax(1)
            total   += tgt.size(0)
            correct += pred.eq(tgt).sum().item()
    return round(loss_sum/len(test_loader),4), round(100.*correct/total,2)

print(f"\n{'='*65}")
print(f"  TEXT EXP2 — Residual Transformer  |  {device.upper()}  |  {EPOCHS} epochs")
print(f"  n_embd={N_EMBD}  n_head={N_HEAD}  n_layer={N_LAYER}  lr={LR}")
print(f"{'='*65}")

for ep in range(EPOCHS):
    t0 = time.time()
    tr  = train_epoch(ep)
    vl, va = evaluate()
    elapsed = time.time() - t0

    history["train_loss"].append(tr["loss"])
    history["train_acc"].append(tr["acc"])
    history["val_loss"].append(vl)
    history["val_acc"].append(va)
    history["sr_overall"].extend(tr["sr_overall"])
    history["sr_steps"].extend(tr["sr_steps"])
    for k, vals in tr["sr_layer"].items():
        history["sr_per_layer"].setdefault(k,[]).extend(vals)

    sr_avg = round(sum(model.sr_per_layer().values())/N_LAYER, 4)
    print(f"Ep {ep+1:02d}/{EPOCHS} | TrLoss={tr['loss']:.4f} TrAcc={tr['acc']:.2f}% "
          f"| ValLoss={vl:.4f} ValAcc={va:.2f}% | SR={sr_avg} | {elapsed:.1f}s")

# ══════════════════════════════════════════════════════════════════════════════
# JSON EXPORT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("JSON OUTPUT — paste into your project as text_exp2_defaults.json")
print("="*65)

output = {
    "trainLoss":  history["train_loss"],
    "valLoss":    history["val_loss"],
    "trainAcc":   history["train_acc"],
    "valAcc":     history["val_acc"],
    "srPerLayer": history["sr_per_layer"],
    "srOverall":  history["sr_overall"],
    "srSteps":    history["sr_steps"],
    "hyperparams": {
        "n_embd": N_EMBD,
        "n_head": N_HEAD,
        "n_layer": N_LAYER,
        "learning_rate": LR,
        "epochs": EPOCHS
    }
}

print(json.dumps(output, indent=2))

# Save JSON file
with open("/kaggle/working/text_exp2_defaults.json","w") as f:
    json.dump(output, f, indent=2)
print("\nSaved to /kaggle/working/text_exp2_defaults.json")

# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING METRICS
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating Plots...")
plt.style.use('dark_background')
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# 1. Loss
axs[0, 0].plot(range(1, EPOCHS+1), history["train_loss"], label="Train Loss", color="#ff716c", linewidth=2)
axs[0, 0].plot(range(1, EPOCHS+1), history["val_loss"], label="Val Loss", color="#ffb74d", linewidth=2)
axs[0, 0].set_title("Loss", fontsize=14, pad=10)
axs[0, 0].set_xlabel("Epochs")
axs[0, 0].set_ylabel("Cross Entropy Loss")
axs[0, 0].grid(True, alpha=0.2)
axs[0, 0].legend()

# 2. Accuracy
axs[0, 1].plot(range(1, EPOCHS+1), history["train_acc"], label="Train Acc", color="#81ecff", linewidth=2)
axs[0, 1].plot(range(1, EPOCHS+1), history["val_acc"], label="Val Acc", color="#4caf50", linewidth=2)
axs[0, 1].set_title("Accuracy", fontsize=14, pad=10)
axs[0, 1].set_xlabel("Epochs")
axs[0, 1].set_ylabel("Accuracy (%)")
axs[0, 1].grid(True, alpha=0.2)
axs[0, 1].legend()

# 3. Stable Rank / Layer
for layer_key, sr_vals in history["sr_per_layer"].items():
    axs[1, 0].plot(history["sr_steps"], sr_vals, label=layer_key, alpha=0.8)
axs[1, 0].set_title("Stable Rank / Layer", fontsize=14, pad=10)
axs[1, 0].set_xlabel("Training Steps")
axs[1, 0].set_ylabel("Stable Rank")
axs[1, 0].grid(True, alpha=0.2)
axs[1, 0].legend()

# 4. Stable Rank Overall
axs[1, 1].plot(history["sr_steps"], history["sr_overall"], color="#a78bfa", linewidth=2)
axs[1, 1].set_title("Stable Rank / Steps", fontsize=14, pad=10)
axs[1, 1].set_xlabel("Training Steps")
axs[1, 1].set_ylabel("Average Stable Rank")
axs[1, 1].grid(True, alpha=0.2)

plt.tight_layout()
plt.show()
