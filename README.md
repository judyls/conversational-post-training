# Conversational Post-Training

Teaching a chatbot to answer the way a person would — short, direct, no padding — and measuring what
that costs.

Ask most AI assistants a simple question and you get five paragraphs, a "Great question!" preamble, and
a summary of what it just told you. This project takes an open model with that habit, tries two standard
methods for training it out, and then checks what else broke in the process. The short version: one
method worked, the other quietly destroyed the model's grasp of facts while the automatic style grader
kept giving it top marks.

**[Try the comparison explorer →](https://judyls.github.io/conversational-post-training/)** — every
answer from all five versions, side by side, written for readers with no background in this. Nothing
runs live; the answers were saved during the test run.

## The question, and the answer

**Question.** Does the more sophisticated training method beat the simple one at teaching conversational
style, and what explains the difference?

**Answer.** No. The simple method won clearly. All three runs of the sophisticated method produced a
model that sounded fine and had lost a third of its factual accuracy — and the style score never
registered it. [Full analysis in WRITE_UP.md →](WRITE_UP.md)

## What was done

The model was given 375 pairs of answers to the same question: one written the way a person texts, one
padded and robotic.

- **Method 1 — imitate.** Show the model only the good answers: *write like these.*
  (Supervised fine-tuning, or SFT.)
- **Method 2 — compare.** Show it both kinds and say *prefer the first.* (Direct Preference
  Optimization, or DPO.) Run three times, letting the model change a lot, a medium amount, or only a
  little.

All five versions — the original plus the four trained ones — then answered the same 95 questions, held
back so none had been seen during training. Three tests scored the answers:

| Test | What it measures | How |
|---|---|---|
| Sounds human | Does it read like a person wrote it? | Claude scores each answer 1–5 |
| Facts right | Did it stay accurate? | TruthfulQA, a public 100-question quiz |
| Useful | Did it actually do the job? | Claude judges each answer yes/no |

## Results

| Version | Run name | Sounds human | Facts right | Useful |
|---|---|---|---|---|
| Original | `base` | 4.13 ± 0.81 | 65% | 88% |
| **Imitate** | `sft` | **4.87 ± 0.34** | **61%** | **84%** |
| Compare · hard | `dpo-beta005` (β=0.05) | 4.78 ± 0.42 | 28% | 42% |
| Compare · medium | `dpo-beta010` (β=0.10) | 4.87 ± 0.34 | 34% | 68% |
| Compare · light | `dpo-beta020` (β=0.20) | 4.89 ± 0.31 | 38% | 76% |

**Imitating good answers cost almost nothing.** Sounding human rose from 4.13 to 4.87 in exchange for
4 points of factual accuracy. It also cut the spread of scores by more than half (0.81 → 0.34), so the
model became reliably short rather than short on average.

**All three comparison runs broke the facts.** Every one held or improved its style score while
factual accuracy fell by 27 to 37 points. The training logs show the model found a shortcut: rather
than working out what made the good answers good, it learned to treat the bad ones as impossible —
which is easy to do and teaches it nothing useful. Once it found that shortcut there was nothing left
to learn from, so training effectively stopped while the model kept drifting away from what it already
knew.

**The damage shows in the writing, not just the scores.** At the hardest setting answers roughly
halved in length — 66 words to 33 on the conversational prompts, 81 to 33 on the task requests — and
turned repetitive. Of those 50 task requests, 28 (56%) open with one of two phrases: "Cut to the
chase..." or "Jump right in...", neither of which the original or the imitation-trained model uses
once. The style grader scored those same answers 4.78 out of 5. One metric was never going to be
enough.

Even the most cautious setting gave up 27 points of accuracy, so none of these versions would be worth
putting in front of real users.

## Technical details

```
generate_dataset.py   →   train_sft.py   →   train_dpo.py   →   eval.py
   375 preference           SFT on chosen        DPO × 3 beta       3-dimension
   pairs (Claude API)       responses only       values (0.05,       eval suite
                                                 0.1, 0.2)
```

**Dataset.** 375 preference pairs generated via the Claude API across three categories: factual Q&A
(30%), conversational (40%), task requests (30%). Each pair has `chosen` = brief and natural,
`rejected` = verbose and padded. Full dataset in `data/preference_pairs.json`.

**Training.**
- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Quantization: 4-bit NF4 via BitsAndBytes + QLoRA
- LoRA: r=16, α=32, target modules q/k/v/o projections (~0.1% of parameters trained)
- SFT: 2 epochs on chosen responses, lr=2e-4
- DPO: 3 epochs from the SFT checkpoint, lr=5e-5, beta ∈ {0.05, 0.1, 0.2}
- Infrastructure: Modal cloud, A10G (24GB VRAM)

**Why SFT before DPO.** DPO works better when the starting policy is already in the target
neighbourhood. Pure DPO from the base model is unstable when the style gap between chosen and rejected
is large. SFT moves the policy toward conversational style first; DPO then refines preferences.

**What beta controls.** Beta is the coefficient on the KL penalty — how far the trained policy may
deviate from the reference model. Lower beta permits more movement, so style transfer is more
aggressive and forgetting is worse. The three runs span the useful range: 0.05 (aggressive), 0.10 (the
DPO paper default), 0.20 (conservative).

**What collapse looked like in the logs.** Loss reached 0.0, grad_norm went to zero, and reward margins
passed 35 where a healthy run sits between 1 and 3. The policy drove rejected-response
log-probabilities toward negative infinity rather than learning a preference that generalises — it
over-optimized a style signal that was too easy to separate.

## Setup

```bash
pip install -r requirements.txt
cp .env.local.example .env.local   # add your ANTHROPIC_API_KEY
```

**Dataset generation** (runs locally, ~$2-3 in API costs):
```bash
python generate_dataset.py --test   # 9 pairs to check quality first
python generate_dataset.py          # full run
```

**Training on Modal** (requires a Modal account + HuggingFace token):
```bash
# One-time setup
modal secret create huggingface-secret HF_TOKEN=hf_...
modal secret create anthropic-secret ANTHROPIC_API_KEY=sk-ant-...
modal volume create conversational-post-training-outputs
modal volume put conversational-post-training-outputs data/preference_pairs.json /data/preference_pairs.json

# SFT (~2-3hrs, ~$3)
modal run modal_train_sft.py

# DPO — run all three betas (~2-3hrs each)
modal run modal_train_dpo.py --beta 0.05
modal run modal_train_dpo.py --beta 0.1
modal run modal_train_dpo.py --beta 0.2

# Eval — all five checkpoints (~4-5hrs)
modal run modal_eval.py
modal volume get conversational-post-training-outputs /results ./results
```

The Modal runners fetch a pinned commit rather than whatever is currently on `main`, because they
execute this repo's code on a GPU with the HuggingFace and Anthropic secrets mounted. Bump `GIT_REF`
at the top of each `modal_*.py` when you want a newer commit to run.

## The comparison explorer

`docs/` is a static page that puts the original model and any trained version side by side on the same
question — the 45 conversational prompts and 50 task requests from the eval, plus all 100 TruthfulQA
questions with per-version right/wrong marks.

Every answer it shows was generated during the eval run and saved; the page runs no model and calls no
API, so it costs nothing to host and cannot drift from the numbers in the write-up.

`docs/index.html` is one self-contained file with the answers embedded, so it opens straight from disk
— no server, no build step, no dependencies. To rebuild it after a new eval run:

```bash
modal volume get conversational-post-training-outputs /results ./results
python build_demo_data.py          # embeds results/*.json into docs/index.html
```

`build_demo_data.py` also derives two style statistics the original eval did not record: mean answer
length, and how concentrated each version's opening phrases are. Those are what surfaced the stock
phrases described above — mode collapse visible in the text rather than in the scores.

## Limitations

1. **The grader is circular.** Claude both generated the training data and judged the output, so the
   style scores measure how well the model learned Claude's taste, not an independent standard of
   "conversational." TruthfulQA is the only test here with a ground truth.
2. **The data is synthetic.** Preference pairs written by a model differ from ones drawn from real user
   conversations. Whether these results hold on real data is untested.
3. **One domain, one size.** 375 prompts and a single 7B model. Nothing here shows whether the findings
   generalise to larger models or to medical, legal or highly technical conversation.
4. **Only 0.1% of the model was trained.** QLoRA adapts a small slice of the network. Conversational
   style is probably spread more widely than that, so full fine-tuning might behave differently —
   possibly with less forgetting, since more of the model participates.

## Dependencies

See `requirements.txt`. Pinned versions are tested on A10G with CUDA 12.4.
