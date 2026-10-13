import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GPTRecConfig:
    vocab_size: int = 50002
    max_len: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(c.n_embd), nn.LayerNorm(c.n_embd)
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd)
        self.proj = nn.Linear(c.n_embd, c.n_embd)
        self.fc = nn.Linear(c.n_embd, 4 * c.n_embd)
        self.out = nn.Linear(4 * c.n_embd, c.n_embd)
        self.n_head, self.p = c.n_head, c.dropout
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = [t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v)]
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, T, C)))
        return x + self.drop(self.out(F.gelu(self.fc(self.ln2(x)))))


class GPTRec(nn.Module):
    def __init__(self, c: GPTRecConfig):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab_size, c.n_embd, padding_idx=0)
        self.pos = nn.Embedding(c.max_len, c.n_embd)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith(("proj.weight", "out.weight")):
                nn.init.normal_(p, 0.0, 0.02 / (2 * c.n_layer) ** 0.5)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def hidden(self, x):
        h = self.drop(self.tok(x) + self.pos(torch.arange(x.shape[1], device=x.device)))
        for b in self.blocks:
            h = b(h)
        return self.ln_f(h)

    def forward(self, x, y=None):
        h = self.hidden(x)
        if y is None:
            return h @ self.tok.weight.T            # weight tying: output = input embeddings
        m = y != -100
        if m.sum() == 0:
            return h.sum() * 0.0
        lg = (h[m] @ self.tok.weight.T).float()      # logits only where a target exists
        lg[:, :2] = float("-inf")                    # PAD and UNK are never predicted
        return F.cross_entropy(lg, y[m])