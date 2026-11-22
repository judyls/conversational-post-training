# Conversational Post-Training

Fine-tuning Qwen 2.5 7B to respond conversationally — brief, natural, non-verbose — without sacrificing factual accuracy or helpfulness.

**Research question:** What preference data and training objective produces an LLM that sounds like a thoughtful person rather than a document generator? Does DPO generalize better than SFT alone for conversational style transfer, and what training dynamics explain the difference?

**Answer:** No — SFT alone is the better intervention at this scale. All three DPO runs collapsed and gave up most of the model's factual accuracy while style scores stayed high. Full analysis in [WRITE_UP.md](WRITE_UP.md).

## Pipeline

```
generate_dataset.py   →   train_sft.py   →   train_dpo.py   →   eval.py
   375 preference           SFT on chosen        DPO × 3 beta       3-dimension
   pairs (Claude API)       responses only       values (0.05,       eval suite
                                                 0.1, 0.2)
```

### Dataset
375 preference pairs generated via Claude API across three categories:
- **Factual Q&A** (30%): questions with unambiguous correct answers
- **Conversational** (40%): casual questions, recommendations, advice
- **Task requests** (30%): write X, explain Y, debug Z

Each pair: `chosen` = brief and natural, `rejected` = verbose and padded.

See `data/preference_pairs.json` for the full dataset.

### Training
- **Base model:** `Qwen/Qwen2.5-7B-Instruct`
- **Quantization:** 4-bit NF4 via BitsAndBytes + QLoRA
- **LoRA:** r=16, α=32, target modules: q/k/v/o projections
- **SFT:** 2 epochs on chosen responses, lr=2e-4
- **DPO:** 3 epochs from SFT checkpoint, lr=5e-5, beta ∈ {0.05, 0.1, 0.2}
- **Infrastructure:** Modal cloud, A10G (24GB VRAM)

### Why SFT before DPO
DPO works better when the starting policy is already in the target neighborhood. Pure DPO from the base model is unstable when the style gap between chosen and rejected is large. SFT moves the policy toward conversational style first; DPO then refines preferences.

### The beta experiment
Beta controls the KL penalty — how far the trained policy can deviate from the reference model. Three runs measure how aggressively you can push style transfer before it hurts accuracy: `dpo-beta005` (0.05, aggressive), `dpo-beta010` (0.10, the DPO paper default), `dpo-beta020` (0.20, conservative).

## Results

Five checkpoints, three evaluation dimensions: conversational quality (Claude Haiku judge, 1–5), factual accuracy (TruthfulQA, 100 multiple-choice questions), helpfulness (Claude Haiku judge, binary). All eval prompts held out from training.

| Checkpoint | Beta | Quality (mean ± std) | TruthfulQA | Helpfulness |
|---|---|---|---|---|
| base | — | 4.13 ± 0.81 | 65.0% | 88.0% |
| **sft** | — | **4.87 ± 0.34** | **61.0%** | **84.0%** |
| dpo-beta005 | 0.05 | 4.78 ± 0.42 | 28.0% | 42.0% |
| dpo-beta010 | 0.10 | 4.87 ± 0.34 | 34.0% | 68.0% |
| dpo-beta020 | 0.20 | 4.89 ± 0.31 | 38.0% | 76.0% |

**SFT is the better intervention:** +0.74 conversational quality for 4pp of TruthfulQA accuracy, and it cut response variance by more than half (std 0.81 → 0.34) — the model became *consistently* brief, not just brief on average.

**All three DPO runs suffered catastrophic forgetting.** Quality held or improved, but TruthfulQA fell 27–37pp. The training logs show why: loss hit 0.0, grad_norm went to zero, and reward margins exceeded 35 (healthy runs sit at 1–3). The model drove rejected-response log-probabilities toward negative infinity instead of learning a generalizable preference — it over-optimized a style signal that was too easy to separate, and lost its grounding in the process.

The beta effect is clean and monotonic: lower beta → more aggressive style transfer → more forgetting. But even the most conservative run (beta=0.2) gave up 27pp of accuracy, which would be unacceptable in a production deployment.

See [WRITE_UP.md](WRITE_UP.md) for the full analysis, including what I would change to make DPO work here.

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

**Training on Modal** (requires Modal account + HuggingFace token):
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

## Limitations

1. **Circular eval:** Claude generates training data and judges outputs — the model learns Claude's aesthetic, not an independently validated notion of "conversational."
2. **Synthetic data only:** Real conversational preference data from logged interactions would be more ecologically valid.
3. **Single domain:** 375 prompts can't cover all conversation types; generalization to medical, legal, or highly technical domains is untested.
4. **QLoRA constraint:** Training ~0.1% of parameters. Conversational style is likely distributed more broadly; full fine-tuning might show different results.

## Dependencies

See `requirements.txt`. Pinned versions are tested on A10G with CUDA 12.4.
