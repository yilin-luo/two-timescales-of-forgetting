"""Decoder-only transformer, adapted from nanoGPT (https://github.com/karpathy/nanoGPT).

Pre-normalisation blocks, learned absolute positions, GELU feed-forward layers of width 4d,
no biases, no dropout, and a linear readout tied to the token embedding.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
                   for t in self.c_attn(x).split(C, dim=2))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd, bias=False)
        self.attn = Attention(n_embd, n_head)
        self.ln_2 = nn.LayerNorm(n_embd, bias=False)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class Transformer(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            token_embedding=nn.Embedding(vocab_size, n_embd),
            positional_embedding=nn.Embedding(block_size, n_embd),
            h=nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)]),
            ln_f=nn.LayerNorm(n_embd, bias=False),
        ))
        self.unembed = nn.Linear(n_embd, vocab_size, bias=False)
        self.transformer.token_embedding.weight = self.unembed.weight       # tied readout

        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):                              # residual projections
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * n_layer))

    def hidden(self, tokens):
        """Representation after the final layer normalisation, shape (batch, length, n_embd)."""
        t = self.transformer
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = t.token_embedding(tokens) + t.positional_embedding(positions)
        for block in t.h:
            x = block(x)
        return t.ln_f(x)

    def forward(self, tokens):
        """Next-token logits at every position, shape (batch, length, vocab_size)."""
        return self.unembed(self.hidden(tokens))

    @torch.no_grad()
    def generate(self, tokens, length):
        """Extend `tokens` of shape (batch, t) to `length` tokens by sampling at temperature 1."""
        while tokens.shape[1] < length:
            logits = self.unembed(self.hidden(tokens)[:, -1])
            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens
