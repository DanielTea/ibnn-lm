# Copyright 2026. Apache License 2.0.
#
# Minimal nanoGPT-style decoder LM. The only non-standard piece is the FFN: set
# ffn="ibnn" to use the IBNN neuron on the MLP hidden layer, or ffn="sm" for an
# ordinary transformer MLP (the Standard Model baseline). Everything else - token and
# positional embeddings, causal self-attention, residuals, LayerNorm, the LM head and the
# next-token cross-entropy objective - is untouched, which is the whole point: it isolates
# the effect of swapping the neuron model while holding parameter count essentially equal
# (IBNN adds exactly one scalar, lambda, per layer).

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import IBNNMLP


@dataclass
class GPTConfig:
    vocab_size: int = 256
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    d_ff: int = None            # defaults to 4 * d_model
    dropout: float = 0.0
    ffn: str = "ibnn"           # "ibnn" or "sm"
    # IBNN hyperparameters (ignored when ffn == "sm")
    ibnn_lambda: float = -0.05
    ibnn_lambda_trainable: bool = True
    ibnn_p: float = 10.0
    ibnn_num_iters: int = 1     # 1 == lite (forward only); >1 unrolls the implicit solve


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.d_model = cfg.d_model
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
                1, 1, cfg.block_size, cfg.block_size),
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.d_model, dim=2)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class StandardMLP(nn.Module):
    """The SM baseline FFN: Linear -> GELU -> Linear."""
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        d_ff = cfg.d_ff or 4 * cfg.d_model
        self.up = nn.Linear(cfg.d_model, d_ff)
        self.down = nn.Linear(d_ff, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.down(F.gelu(self.up(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        if cfg.ffn == "ibnn":
            self.mlp = IBNNMLP(
                cfg.d_model, d_ff=cfg.d_ff, dropout=cfg.dropout,
                lam=cfg.ibnn_lambda, lam_trainable=cfg.ibnn_lambda_trainable,
                p=cfg.ibnn_p, num_iters=cfg.ibnn_num_iters, activation="gelu",
            )
        elif cfg.ffn == "sm":
            self.mlp = StandardMLP(cfg)
        else:
            raise ValueError(f"unknown ffn {cfg.ffn}")

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


def copy_sm_weights_into_ibnn(sm_model: GPT, ibnn_model: GPT):
    """Warm-start an IBNN model from a trained SM model of identical shape.

    This is the LM analog of the paper's surrogate warmup: train a normal transformer, then
    transfer every matching parameter (embeddings, attention, layernorms, head, and the FFN
    up/down projections) into the IBNN model and switch lambda on. Only lambda has no source
    and keeps its init.
    """
    sm_sd = sm_model.state_dict()
    tgt_sd = ibnn_model.state_dict()
    copied, skipped = 0, []
    for k, v in tgt_sd.items():
        # FFN keys differ: IBNN up-projection lives at ...mlp.up.weight/.bias (IBNNLinear),
        # SM up lives at ...mlp.up.weight/.bias (nn.Linear) - same names, same shapes.
        if k in sm_sd and sm_sd[k].shape == v.shape:
            tgt_sd[k] = sm_sd[k].clone()
            copied += 1
        else:
            skipped.append(k)
    ibnn_model.load_state_dict(tgt_sd, strict=False)
    return copied, skipped
