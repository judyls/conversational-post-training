# Conversational Post-Training: Findings

**Research question:** What preference data and training objective produces an LLM that responds conversationally — brief, natural, direct — without sacrificing factual accuracy or helpfulness?

---

## Results

| Checkpoint | Quality (mean ± std) | TruthfulQA Accuracy | Helpfulness Rate |
|---|---|---|---|
| base | 4.13 ± 0.81 | 65.0% | 88.0% |
| sft | 4.87 ± 0.34 | 61.0% | 84.0% |
| dpo-005 | 4.78 ± 0.42 | 28.0% | 42.0% |
| dpo-010 | 4.87 ± 0.34 | 34.0% | 68.0% |
| dpo-020 | 4.89 ± 0.31 | 38.0% | 76.0% |

Five checkpoints evaluated across three dimensions: conversational quality (Claude Haiku judge, 1–5), factual accuracy (TruthfulQA multiple-choice), helpfulness (Claude Haiku judge, binary). 50 quality prompts, 100 TruthfulQA questions, 50 helpfulness prompts — all held out from training data.

---

## Finding 1: SFT is the better intervention

SFT improved conversational quality from 4.13 to 4.87 (+0.74) while dropping TruthfulQA accuracy by only 4pp (65% → 61%) and helpfulness by 4pp (88% → 84%). That's a favorable tradeoff: large style improvement, small capability cost.

The variance reduction is also notable — std dropped from 0.81 to 0.34. SFT didn't just raise the average quality; it made the model *consistently* more conversational, reducing the frequency of verbose outlier responses.

## Finding 2: DPO caused catastrophic forgetting

All three DPO runs maintained or slightly improved quality scores but caused severe factual degradation:

- dpo-005: TruthfulQA dropped to 28% — a 37pp loss from base. Helpfulness fell to 42%.
- dpo-010: 34% accuracy, 68% helpfulness.
- dpo-020: 38% accuracy, 76% helpfulness.

Beta is a real knob. Lower beta → more aggressive style transfer → more forgetting. The pattern is clean and monotonic across all three evaluation dimensions.

## Finding 3: DPO collapse explains the degradation

Training logs showed the DPO runs collapsed: loss hit 0.0, grad_norm approached zero, reward margins exceeded 35 (healthy runs typically see 1–3). Reward accuracy hit 1.0 early and stayed there.

DPO collapse happens when the model takes the path of least resistance — driving the rejected response log-probability toward negative infinity rather than genuinely improving on chosen responses. The DPO loss reaches zero, but the model has overfit the style signal rather than learning a generalizable preference. This manifests directly in eval: quality scores stay high (the model sounds conversational) but factual accuracy collapses (the model has lost its grounding).

The cause here is a combination of low beta values and a strong signal-to-noise ratio in the training data. With 375 preference pairs where the style difference between chosen and rejected is stark and consistent, the model can easily separate them — no subtle reasoning required. There's nothing stopping it from over-optimizing.

## Finding 4: The answer to the research question

DPO does not generalize better than SFT alone for conversational style transfer at this scale.

SFT on chosen responses is the more efficient and less destructive intervention. It captures most of the quality improvement with a fraction of the capability cost. DPO as implemented here is over-kill for a style task: it pushes too hard on the preference signal and the factual accuracy penalty is severe.

The optimal checkpoint from this experiment is SFT. Among DPO runs, dpo-020 offers the best tradeoff, but it still drops TruthfulQA accuracy by 27pp — a cost that would be unacceptable in a production deployment.

---

## What this suggests for future work

**Data quality over quantity for DPO.** The signal here was too clean. Real preference data with more ambiguous pairs — where the style difference is smaller and the factual content matters more — would force DPO to make finer-grained distinctions rather than collapsing on a simple pattern.

**Higher beta or beta scheduling.** Starting with high beta and annealing down across training would let the model learn from the preference signal without the early collapse that zeroes out gradients before training is meaningful.

**Early stopping on reward margin.** Monitoring `rewards/margins` and stopping when it exceeds a threshold (e.g., 5–10) would prevent the collapse. The current runs had no such guard.

**Combining SFT and DPO differently.** SFT for 2 epochs followed by light DPO (1 epoch, beta=0.2) might capture the variance reduction benefit of DPO without the catastrophic forgetting. The SFT model already sits in a good neighborhood; a conservative DPO pass might refine it without destroying factual accuracy.

---

## Limitations

1. **Circular eval:** Claude generates training data and judges outputs. Quality and helpfulness scores measure how well the model learned Claude's aesthetic, not an independently validated notion of "conversational." TruthfulQA is the only dimension with a ground-truth signal.

2. **Synthetic data only:** Preference pairs generated by a language model differ from those derived from real user interactions. Ecological validity is untested.

3. **QLoRA constraint:** Training ~0.1% of parameters. Conversational style is likely distributed more broadly across the network; full fine-tuning might behave differently, potentially showing less catastrophic forgetting because more of the model participates in the update.

4. **Single domain and scale:** 375 prompts, one 7B model. Whether these findings generalize to larger models or more diverse prompt distributions is an open question.
