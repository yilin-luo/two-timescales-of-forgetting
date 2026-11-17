# Two Timescales of Forgetting

Code for the main experiment of

> **Two Timescales of Forgetting in Fine-tuning of Language Models: Fast Behavioral Suppression and Slow Representational Drift**

A transformer is pretrained on a synthetic hierarchical language, the Random Hierarchy Model
(RHM), and then fine-tuned on a restricted version of the same language in which half of the
production rules are deleted. During fine-tuning, the network stops *generating* the deleted
rules quickly, level by level from the bottom of the hierarchy to the root. Its internal
*representation* of the deleted rules, and its ability to use them when they appear in the
context, persist for much longer.

**Scope.** This repository reproduces the main result on the Random Hierarchy Model
(Figure 2 of the paper). The natural-language experiments and the additional analyses of the
paper are not included.

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.11 and PyTorch 2.6. A single GPU is sufficient.

## Reproducing Figure 2

```bash
python train.py pretrain --seed 0    # pretrain on the full grammar
python train.py finetune --seed 0    # fine-tune on the restricted grammar, saving checkpoints
python evaluate.py --seed 0          # measure every checkpoint
python plot.py                       # plot the mean and standard deviation over all seeds found
```

The paper uses the seeds 0, 1, 2 and 3. Everything is written under `runs/seed<seed>/`:

| File | Content |
|---|---|
| `pretrain/rules.pt` | the pretraining grammar |
| `pretrain/model.pt` | the pretrained weights |
| `finetune/rules.pt` | the fine-tuning grammar |
| `finetune/checkpoints/step_<t>.pt` | weights after `t` fine-tuning steps, on a logarithmic grid |
| `*/trace.json` | test loss along training |
| `evaluation.json` | the measurements below, for every checkpoint |

All settings are in [`config.yaml`](config.yaml).

## What is measured

Sequences have 16 tokens and a latent binary tree with four levels above the tokens. The
tokens are level 0 and the root is level 4. The pretraining grammar has 6 rules (synonyms) for
every symbol at every level. The fine-tuning grammar keeps 3 of them and deletes the others.

| Panel | Measurement | Code |
|---|---|---|
| (a) | **Deleted-rule rate.** The network generates sequences, which are parsed with the pretraining grammar. The rate is the fraction of nodes at each level that use a deleted rule. Its value under the pretraining grammar is 1/2. | `deleted_rule_rate` in `evaluate.py` |
| (b) | **Deleted-rule overlap** $q^R$. A sequence of the fine-tuning grammar is compared with a copy in which one latent node uses a deleted synonym of the same symbol. $q^R$ is the overlap of the two standardised hidden representations at the second-to-last token. It is close to 1 while the network represents the deleted synonym like the kept ones. The *symbol reference* is the same overlap when the symbol of the node is replaced by a different one. | `probe_inputs`, `overlap` in `evaluate.py` |
| (c) | **Last-token loss on three test families.** *Kept grammar*: every node uses a kept rule. *Deleted grammar*: every node uses a deleted rule. *Deleted grammar in context*: the ancestors of the last token use kept rules and all other nodes use deleted rules. The dashed line is the Bayes-optimal loss of the fine-tuning grammar. | `test_families`, `last_token_loss` in `evaluate.py`; `bayes_last_token_loss` in `rhm.py` |

## Files

| File | Content |
|---|---|
| `rhm.py` | the grammar: drawing rules, deleting rules, sampling, parsing, Bayes-optimal last-token loss |
| `model.py` | the transformer |
| `train.py` | pretraining and fine-tuning |
| `evaluate.py` | the three measurements |
| `plot.py` | the figure |

## Notes

- **Fine-tuning length.** `finetune_steps` is the nominal length that defines the checkpoint
  grid. Checkpoints are written during training and `evaluate.py` uses the ones it finds, so a
  run can be stopped at any time. The curves of Figure 2 end near $5\times10^6$ steps.
- **Cost.** A checkpoint takes about 50 MB, and a full fine-tuning run saves about 250 of them.
  On an older GPU (Quadro P5000), training runs at about 20 steps per second and evaluating one
  checkpoint takes about 10 seconds.

## Acknowledgements

The Random Hierarchy Model code builds on the implementation by Antonio Sclocchi and
Alessandro Favero. The transformer is adapted from [nanoGPT](https://github.com/karpathy/nanoGPT)
by Andrej Karpathy.

## License

MIT, see [`LICENSE`](LICENSE).
