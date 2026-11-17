"""Evaluate every fine-tuning checkpoint of one seed: the three panels of Figure 2.

    python evaluate.py --seed 0     # reads runs/seed0, writes runs/seed0/evaluation.json

(a) generation probe:      fraction of nodes that use a deleted rule in generated sequences;
(b) representation probe:  overlap q^R of the hidden representation when a kept rule in the
                           context is replaced by a deleted synonym, and its symbol reference;
(c) prediction loss:       last-token loss on three test families, and the Bayes-optimal loss
                           of the fine-tuning grammar.
"""
import argparse
import glob
import json
import os
import re

import torch
import torch.nn.functional as F
import yaml

import rhm
from model import Transformer

NUM_GENERATED = 8192        # generated sequences per checkpoint
NUM_PROBE = 4096            # original sequences of the representation probe
NUM_FAMILY = 8192           # sequences per test family
FAMILIES = ("kept grammar", "deleted grammar", "deleted grammar in context")


def batched(f, x, size=4096):
    return torch.cat([f(chunk) for chunk in x.split(size)])


# ---- (a) generation probe ---------------------------------------------------------------------

def deleted_rule_rate(model, rules, m_keep, first_tokens):
    """The model continues each first token at temperature 1; the sequences are parsed with the
    pretraining grammar. Returns, for every level, the fraction of all nodes that use a deleted
    rule. Invalid nodes stay in the denominator and count as neither kept nor deleted."""
    L = len(rules)
    _, m, s = rules[L].shape
    sequences = model.generate(first_tokens[:, None], s ** L).cpu()
    _, synonym = rhm.parse(sequences, rules)
    return {l: ((synonym[l] >= 0) & (synonym[l] < m - m_keep)).float().mean().item()
            for l in synonym}


# ---- (b) representation probe -----------------------------------------------------------------

def probe_inputs(rules, m_keep, n, generator):
    """Inputs of the representation probe.

    x holds sequences of the fine-tuning grammar. The probed node is the second from the right
    at level l = 1..L-1; its subtree contains neither the second-to-last token nor the last one.
      deleted[l]: x with the rule of that node replaced by a deleted synonym of the same symbol;
      other[l]:   x with the symbol of that node replaced by a different one (symbol reference).
    In both, the descendants of the node are regenerated with kept rules.
    """
    L = len(rules)
    v, m, s = rules[L].shape
    kept = rhm.restrict(rules, m_keep)
    x = rhm.sample(kept, n, generator)
    symbol, _ = rhm.parse(x, rules)
    deleted, other = {}, {}
    for l in range(1, L):
        i = symbol[l].shape[1] - 2
        a, span = symbol[l][:, i], slice(i * s ** l, (i + 1) * s ** l)
        k = torch.randint(0, m - m_keep, (n,), generator=generator)
        deleted[l] = x.clone()
        deleted[l][:, span] = rhm.expand(kept, rules[l][a, k], l - 1, generator)
        b = (a + torch.randint(1, v, (n,), generator=generator)) % v
        other[l] = x.clone()
        other[l][:, span] = rhm.expand(kept, b[:, None], l, generator)

        # check: exactly the probed node uses a deleted rule, and its symbol is unchanged
        sym, syn = rhm.parse(deleted[l], rules)
        used = torch.cat([syn[j] for j in syn], dim=1)
        assert (used >= 0).all() and (used < m - m_keep).sum(1).eq(1).all()
        assert (syn[l][:, i] < m - m_keep).all()
        assert all(torch.equal(sym[j], symbol[j]) for j in range(l, L + 1))
    return x, deleted, other


def overlap(h, h_transformed):
    """q = E[h(x) . h(Rx)] / d, with both representations standardised coordinate-wise by the
    mean and standard deviation of the untransformed inputs."""
    mean, std = h.mean(0), h.std(0).clamp_min(1e-6)
    return (((h - mean) / std) * ((h_transformed - mean) / std)).mean().item()


# ---- (c) prediction loss on the three test families -------------------------------------------

def test_families(rules, m_keep, n, generator):
    """Sequences of the pretraining grammar in which every node uses a kept rule; every node
    uses a deleted rule; or every node uses a deleted rule except the ancestors of the last
    token (the rightmost node of each level), which use kept rules."""
    L = len(rules)
    s = rules[L].shape[2]
    nowhere = {l: torch.zeros(s ** (L - l), dtype=torch.bool) for l in rules}
    everywhere = {l: ~nowhere[l] for l in rules}
    context = {l: torch.arange(s ** (L - l)) < s ** (L - l) - 1 for l in rules}
    return {name: rhm.sample_with_deleted_rules(rules, m_keep, where, n, generator)
            for name, where in zip(FAMILIES, (nowhere, everywhere, context))}


def last_token_loss(model, x):
    logits = batched(lambda chunk: model(chunk[:, :-1])[:, -1], x)
    return F.cross_entropy(logits, x[:, -1]).item()


# -----------------------------------------------------------------------------------------------

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="seed of the grammar")
    parser.add_argument("--runs", default="runs", help="root directory of the runs")
    args = parser.parse_args()
    run = os.path.join(args.runs, f"seed{args.seed}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(run, "finetune", "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    rules = torch.load(os.path.join(run, "pretrain", "rules.pt"))       # pretraining grammar
    m, m_keep, L = cfg["num_synonyms"], cfg["num_kept"], cfg["num_levels"]
    kept = rhm.restrict(rules, m_keep)                                  # fine-tuning grammar

    generator = torch.Generator().manual_seed(0)
    first_tokens = rhm.sample(kept, NUM_GENERATED, generator)[:, 0].to(device)
    x, deleted, other = probe_inputs(rules, m_keep, NUM_PROBE, generator)
    families = {name: t.to(device) for name, t in
                test_families(rules, m_keep, NUM_FAMILY, generator).items()}
    bayes_loss = rhm.bayes_last_token_loss(kept, rhm.sample(kept, NUM_FAMILY, generator))

    out = {"steps": [],
           "deleted_rule_rate": {l: [] for l in range(1, L + 1)},
           "reference_rate": 1 - m_keep / m,                           # under the pretraining grammar
           "overlap": {l: [] for l in range(1, L)},
           "symbol_reference": {l: [] for l in range(1, L)},
           "loss": {name: [] for name in FAMILIES},
           "bayes_loss": bayes_loss}

    model = Transformer(cfg["vocab_size"], cfg["branching"] ** L,
                        cfg["n_layer"], cfg["n_head"], cfg["n_embd"]).to(device)

    def representation(tokens):         # after the final normalisation, at the second-to-last token
        return batched(lambda chunk: model.hidden(chunk[:, :-1].to(device))[:, -1], tokens)

    def step_of(path):
        return int(re.search(r"step_(\d+)\.pt$", path).group(1))

    for path in sorted(glob.glob(os.path.join(run, "finetune", "checkpoints", "step_*.pt")), key=step_of):
        step = step_of(path)
        model.load_state_dict(torch.load(path, map_location=device))
        out["steps"].append(step)

        torch.manual_seed(step)
        for l, rate in deleted_rule_rate(model, rules, m_keep, first_tokens).items():
            out["deleted_rule_rate"][l].append(rate)

        h = representation(x)
        for l in range(1, L):
            out["overlap"][l].append(overlap(h, representation(deleted[l])))
            out["symbol_reference"][l].append(overlap(h, representation(other[l])))

        for name in FAMILIES:
            out["loss"][name].append(last_token_loss(model, families[name]))

        print(f"step {step:>9d}   deleted-rule rate "
              + " ".join(f"{out['deleted_rule_rate'][l][-1]:.3f}" for l in range(1, L + 1))
              + "   q^R " + " ".join(f"{out['overlap'][l][-1]:.3f}" for l in range(1, L))
              + "   loss " + " ".join(f"{out['loss'][name][-1]:.3f}" for name in FAMILIES), flush=True)

    with open(os.path.join(run, "evaluation.json"), "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
