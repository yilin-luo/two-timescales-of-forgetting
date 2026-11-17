"""The Random Hierarchy Model (RHM): a probabilistic context-free grammar on a regular tree.

Levels follow the paper: the tokens are level 0 and the root is level L. A grammar is a dict
`rules`. `rules[l]`, for l = 1..L, is the rule book that expands a symbol of level l into s
symbols of level l-1. It has shape (v, m, s): `rules[l][a, k]` is the k-th synonym of symbol a.
"""
import random
from itertools import product

import torch


def sample_rules(v, m, s, L, seed):
    """Draw a grammar: every rule book holds v*m distinct s-tuples, m for each of the v symbols.

    Python's `random.sample` may differ between Python versions, so a grammar is drawn once,
    at pretraining, and saved; every later step loads the saved file.
    """
    random.seed(seed)
    tuples = list(product(range(v), repeat=s))
    return {l: torch.tensor(random.sample(tuples, v * m)).reshape(v, m, s) for l in range(L, 0, -1)}


def restrict(rules, m_keep):
    """The fine-tuning grammar: keep the last m_keep synonyms of every symbol in every rule book."""
    return {l: r[:, r.shape[1] - m_keep:].clone() for l, r in rules.items()}


def expand(rules, x, level, generator):
    """Expand symbols x of shape (n, w) at `level` down to tokens, with uniformly drawn synonyms."""
    for l in range(level, 0, -1):
        k = torch.randint(0, rules[l].shape[1], x.shape, generator=generator)
        x = rules[l][x, k].flatten(1)
    return x


def sample(rules, n, generator):
    """n token sequences of shape (n, s^L): uniform root symbol, uniform synonym at every node."""
    L = len(rules)
    root = torch.randint(0, rules[L].shape[0], (n, 1), generator=generator)
    return expand(rules, root, L, generator)


def sample_with_deleted_rules(rules, m_keep, deleted, n, generator):
    """n sequences of the full grammar with a prescribed kind of rule at every node.

    `deleted[l]` is a boolean tensor over the nodes of level l, from left to right. A node uses
    a deleted rule (one of the first m - m_keep synonyms) where it is True, a kept rule elsewhere.
    """
    L = len(rules)
    v, m, _ = rules[L].shape
    x = torch.randint(0, v, (n, 1), generator=generator)
    for l in range(L, 0, -1):
        k_deleted = torch.randint(0, m - m_keep, x.shape, generator=generator)
        k_kept = torch.randint(m - m_keep, m, x.shape, generator=generator)
        x = rules[l][x, torch.where(deleted[l], k_deleted, k_kept)].flatten(1)
    return x


def parse(tokens, rules):
    """Parse sequences bottom-up with the grammar.

    No s-tuple belongs to two symbols of the same rule book, so the latent tree of a sequence is
    unique. Returns two dicts over the levels l = 1..L, each holding a tensor of shape
    (n, s^(L-l)): the symbol of every node, and the index of the synonym it uses. Both are -1 at
    invalid nodes, where the children are not a rule of the grammar or a node below is invalid.
    """
    L = len(rules)
    v, _, s = rules[L].shape
    base = v ** torch.arange(s - 1, -1, -1)             # an s-tuple of symbols -> one integer
    x, symbol, synonym = tokens, {}, {}
    for l in range(1, L + 1):
        codes = (rules[l] * base).sum(-1)               # (v, m)
        table = torch.full((2, v ** s), -1)
        table[0, codes] = torch.arange(v)[:, None].expand_as(codes)
        table[1, codes] = torch.arange(codes.shape[1]).expand_as(codes)
        children = x.reshape(len(x), -1, s)
        valid = (children >= 0).all(-1)
        found = table[:, (children.clamp(min=0) * base).sum(-1)]
        symbol[l], synonym[l] = torch.where(valid, found, -1)
        x = symbol[l]
    return symbol, synonym


def bayes_last_token_loss(rules, tokens):
    """Bayes-optimal cross-entropy of the last token given all the preceding ones.

    `tokens` are sequences of the grammar. The subtrees to the left of the last token's
    ancestors are fully observed, so parsing gives their symbols. The posterior of the last
    token is then a sum over the symbols of its L ancestors, computed from the root down.
    """
    L = len(rules)
    v, _, s = rules[L].shape
    base = v ** torch.arange(s - 2, -1, -1)
    symbol, _ = parse(tokens, rules)
    symbol[0] = tokens
    belief = torch.ones(len(tokens), v, dtype=torch.float64)         # over the root symbol
    for l in range(L, 0, -1):
        # allowed[a, c, b] = 1 if symbol a has a rule whose first s-1 children are c and last is b
        allowed = torch.zeros(v, v ** (s - 1), v, dtype=torch.float64)
        allowed[torch.arange(v)[:, None], (rules[l][..., :-1] * base).sum(-1), rules[l][..., -1]] = 1
        siblings = (symbol[l - 1][:, -s:-1] * base).sum(-1)          # left siblings of the last node
        belief = torch.einsum("na,anb->nb", belief, allowed[:, siblings])
    p = belief / belief.sum(1, keepdim=True)
    return -p[torch.arange(len(tokens)), tokens[:, -1]].log().mean().item()
