import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gpt_rec import Block, GPTRec, GPTRecConfig


class RewardModel(nn.Module):
    """Predicts [ratio, completed, like, dislike] logits for the event at each position."""

    def __init__(self, c: GPTRecConfig, tok_loglen: torch.Tensor):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab_size, c.n_embd, padding_idx=0)
        self.pos = nn.Embedding(c.max_len, c.n_embd)
        self.fb_proj = nn.Linear(4, c.n_embd)       # feedback of the previous event
        self.len_proj = nn.Linear(1, c.n_embd)      # standardized log track length
        self.register_buffer("tok_loglen", tok_loglen)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.head = nn.Linear(c.n_embd, 4)
        self.apply(GPTRec._init)

    def forward(self, x, fb_prev):
        T = x.shape[1]
        h = (self.tok(x) + self.pos(torch.arange(T, device=x.device))
             + self.fb_proj(fb_prev) + self.len_proj(self.tok_loglen[x].unsqueeze(-1)))
        h = self.drop(h)
        for b in self.blocks:
            h = b(h)
        return self.head(self.ln_f(h))              # (B, T, 4) logits


def rm_loss(out, y_ratio, y_like, y_dis, mask):
    o = out[mask]
    r = y_ratio[mask]
    l_ratio = F.binary_cross_entropy_with_logits(o[:, 0], r)
    l_comp = F.binary_cross_entropy_with_logits(o[:, 1], (r >= 0.5).float())
    l_like = F.binary_cross_entropy_with_logits(o[:, 2], y_like[mask])
    l_dis = F.binary_cross_entropy_with_logits(o[:, 3], y_dis[mask])
    return l_ratio + l_comp + 0.5 * l_like + 0.5 * l_dis, (l_ratio.item(), l_comp.item(), l_like.item(), l_dis.item())