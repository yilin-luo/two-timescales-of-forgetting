"""Pretrain a transformer on a Random Hierarchy Model, or fine-tune it on the restricted grammar.

    python train.py pretrain --seed 0     # writes runs/seed0/pretrain
    python train.py finetune --seed 0     # reads runs/seed0/pretrain, writes runs/seed0/finetune

Both stages minimise the next-token cross-entropy on freshly sampled sequences. Fine-tuning
starts from the pretrained weights with a new optimiser and saves the weights on a logarithmic
grid of steps, for evaluate.py.
"""
import argparse
import json
import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import rhm
from model import Transformer


def log_grid(max_steps, num):
    """Nominal `num`-point logarithmic grid from 1 to max_steps, as distinct integers."""
    return sorted(set(np.logspace(0, np.log10(max_steps), num).astype(int).tolist()) | {max_steps})


@torch.no_grad()
def test_loss(model, test, batch_size=8192):
    """Cross-entropy at each predicted position, averaged over the test set."""
    ce = [F.cross_entropy(model(x[:, :-1]).transpose(1, 2), x[:, 1:], reduction="none")
          for x in test.split(batch_size)]
    return torch.cat(ce).mean(0)


def train(model, rules, steps, cfg, out, save_checkpoints):
    device = next(model.parameters()).device
    data = torch.Generator().manual_seed(cfg["data_seed"])
    test = rhm.sample(rules, cfg["test_size"], torch.Generator().manual_seed(cfg["data_seed"] + 999))
    test = test.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"],
                                 fused=device.type == "cuda")
    grid = set(log_grid(steps, cfg["num_checkpoints"])) | {0}
    trace = {"steps": [], "loss": [], "last_token_loss": []}
    if save_checkpoints:
        os.makedirs(os.path.join(out, "checkpoints"))

    for step in range(steps + 1):
        if step > 0:
            x = rhm.sample(rules, cfg["batch_size"], data).to(device)
            logits = model(x[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if step in grid:
            ce = test_loss(model, test)
            trace["steps"].append(step)
            trace["loss"].append(ce.mean().item())              # mean over the predicted positions
            trace["last_token_loss"].append(ce[-1].item())
            print(f"step {step:>9d}   test loss {ce.mean():.4f}   last token {ce[-1]:.4f}", flush=True)
            with open(os.path.join(out, "trace.json"), "w") as f:
                json.dump(trace, f)
            if save_checkpoints:
                torch.save(model.state_dict(), os.path.join(out, "checkpoints", f"step_{step}.pt"))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["pretrain", "finetune"])
    parser.add_argument("--seed", type=int, default=0, help="seed of the grammar")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--runs", default="runs", help="root directory of the runs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True                # faster matrix products on GPU
    pretrain_dir = os.path.join(args.runs, f"seed{args.seed}", "pretrain")
    out = os.path.join(args.runs, f"seed{args.seed}", args.stage)
    os.makedirs(out)                                            # never overwrites an existing run
    shutil.copy(args.config, os.path.join(out, "config.yaml"))

    torch.manual_seed(cfg["init_seed"])
    model = Transformer(cfg["vocab_size"], cfg["branching"] ** cfg["num_levels"],
                        cfg["n_layer"], cfg["n_head"], cfg["n_embd"]).to(device)

    if args.stage == "pretrain":
        rules = rhm.sample_rules(cfg["vocab_size"], cfg["num_synonyms"], cfg["branching"],
                                 cfg["num_levels"], args.seed)
        torch.save(rules, os.path.join(out, "rules.pt"))
        train(model, rules, cfg["pretrain_steps"], cfg, out, save_checkpoints=False)
        torch.save(model.state_dict(), os.path.join(out, "model.pt"))
    else:
        rules = rhm.restrict(torch.load(os.path.join(pretrain_dir, "rules.pt")), cfg["num_kept"])
        torch.save(rules, os.path.join(out, "rules.pt"))
        model.load_state_dict(torch.load(os.path.join(pretrain_dir, "model.pt"), map_location=device))
        train(model, rules, cfg["finetune_steps"], cfg, out, save_checkpoints=True)


if __name__ == "__main__":
    main()
