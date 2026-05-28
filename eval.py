"""
Conversational Post-Training — Evaluation Suite
Three dimensions, five checkpoints: base model, SFT, DPO beta=0.05/0.1/0.2.

Eval dimensions:
  1. Conversational quality  — Claude Haiku judges naturalness/conciseness (1–5)
  2. Factual accuracy        — TruthfulQA multiple-choice
  3. Helpfulness             — Claude Haiku judges whether response actually helped (binary)

Usage:
    export ANTHROPIC_API_KEY=your_key

    # Evaluate base model only
    python eval.py --base

    # Evaluate one adapter
    python eval.py --adapter sft:outputs/sft-run/final

    # Evaluate all checkpoints (base is always included)
    python eval.py \\
        --adapter sft:outputs/sft-run/final \\
        --adapter dpo-005:outputs/dpo-beta005/final \\
        --adapter dpo-010:outputs/dpo-beta010/final \\
        --adapter dpo-020:outputs/dpo-beta020/final

    # Run only specific evals
    python eval.py --adapter sft:outputs/sft-run/final --eval quality
    python eval.py --adapter sft:outputs/sft-run/final --eval truthfulqa
    python eval.py --adapter sft:outputs/sft-run/final --eval helpfulness
"""

import json
import time
import argparse
import os
from pathlib import Path

import torch
import anthropic
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset as hf_load_dataset
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env.local")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
JUDGE_MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = (
    "You are a helpful assistant. Be direct, concise, and natural. "
    "Get to the point without filler or padding."
)

# ── Held-out eval prompts ─────────────────────────────────────────────────────
# These are NOT in the training data. Do not add training prompts here.

QUALITY_EVAL_PROMPTS = [
    # Factual
    "What's the difference between RAM and storage?",
    "How does HTTPS keep my connection secure?",
    "What actually happens during sleep?",
    "Why does the sky turn red at sunset?",
    "What's the difference between a virus and a bacterium?",
    "How do noise-canceling headphones work?",
    "What is compound interest and why does it matter?",
    "Why do we dream?",
    "What's the difference between machine learning and AI?",
    "How does a vaccine work?",
    "What causes inflation?",
    "What is the difference between a stock and a bond?",
    "How does GPS know where I am?",
    "What's the difference between type 1 and type 2 diabetes?",
    "Why is the ocean salty?",
    # Conversational
    "I have an hour to kill at the airport. What should I do?",
    "Should I learn Python or JavaScript first?",
    "I can't stop procrastinating on this project. Any advice?",
    "What's a good gift for someone who says they don't want anything?",
    "I'm burnt out but can't take time off. What helps?",
    "Is it worth paying for a gym membership or just work out at home?",
    "I keep forgetting to drink water. How do I fix this?",
    "Any tips for staying focused when working from home?",
    "I want to start reading more but I always fall asleep after two pages.",
    "Should I negotiate my job offer or just accept it?",
    "My code review has a lot of comments. Is that bad?",
    "I have a 45-minute lunch break. What's a good way to decompress?",
    "My manager never gives feedback. Is that normal?",
    "I want to learn to cook but don't know where to start.",
    "Is it worth learning vim?",
    "What's the fastest way to improve at chess?",
    "I keep waking up at 3am and can't get back to sleep.",
    "Should I use tabs or spaces?",
    "My apartment is always cluttered. What's the easiest way to fix that?",
    "I have a phone interview tomorrow. What should I prepare?",
    # Task requests
    "Help me write a subject line for a cold email to a recruiter.",
    "Explain what a p-value is like I'm not a statistician.",
    "Write a short out-of-office message for a two-week vacation.",
    "Give me a one-paragraph summary of what Docker is and why it exists.",
    "Help me write a polite follow-up email after a job interview.",
    "Explain what a REST API is without using jargon.",
    "Give me three ways to make this sentence less passive: 'The report was completed by the team.'",
    "Help me write a LinkedIn connection request to someone I met at a conference.",
    "Explain the difference between async and sync programming simply.",
    "Write a one-sentence description of my job for a dinner party: I'm a backend engineer at a fintech startup.",
]

HELPFULNESS_EVAL_PROMPTS = [
    "Help me write a two-sentence bio for my GitHub profile. I'm a senior ML engineer.",
    "I need to explain to my manager why this bug fix took three days. Help me draft that.",
    "Write a Slack message telling my team our launch is delayed by a week.",
    "Help me write a cover letter opening paragraph for a machine learning role.",
    "I need to decline a meeting invite politely. Write the message.",
    "Help me give constructive feedback on a junior engineer's code review.",
    "Write a one-paragraph project update for a stakeholder who isn't technical.",
    "I need to ask my landlord to fix the heat. Write the email.",
    "Help me write an apology email for missing a deadline.",
    "Write a short agenda for a 30-minute team retrospective.",
    "Explain what this regex does: ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$",
    "I need to summarize a 10-page research paper in three bullet points. How should I approach it?",
    "Help me write interview questions for a junior data scientist role.",
    "Write a README section explaining how to run the tests for a Python project.",
    "I need to negotiate a higher salary. Give me an opening line.",
    "Help me write a one-sentence description of what gradient descent does.",
    "Write a commit message for a change that refactors the authentication module.",
    "Help me structure a 5-minute presentation about a project I shipped.",
    "I need to tell a candidate we're not moving forward. Write the rejection email.",
    "Help me write a comment explaining a complex SQL join to a future reader.",
    "Give me a template for a weekly status update email.",
    "Help me write a bug report for a race condition I found in production.",
    "I need to ask a colleague for a favor without sounding pushy. Draft the message.",
    "Write a short description of the SFT training phase for a technical blog post.",
    "Help me explain to a non-technical stakeholder why we can't just 'add the feature quickly.'",
    "Write a performance review self-assessment bullet point for shipping a new API.",
    "Help me write a clear error message for when a user uploads a file that's too large.",
    "I need to onboard a new teammate to a complex codebase. What should I include in the intro doc?",
    "Write a tweet-length description of what DPO is.",
    "Help me write a one-paragraph abstract for a technical blog post about this fine-tuning project.",
    "I need to explain what a transformer is to a product manager. Keep it under 3 sentences.",
    "Help me write a polite but firm response to a colleague who keeps interrupting me in meetings.",
    "Write a short description of what this project does for my resume.",
    "I need to ask for a deadline extension. Write the email.",
    "Help me write a data card for a synthetic preference dataset.",
    "Give me a template for documenting an API endpoint.",
    "Help me write a one-sentence hypothesis for an A/B test.",
    "I need to write a postmortem for a production incident. What sections should it have?",
    "Help me explain the difference between precision and recall to a business stakeholder.",
    "Write a short message thanking a mentor for their time.",
    "I need to turn three pages of meeting notes into a one-paragraph summary.",
    "Help me write a job description bullet point for a role that requires experience with LLMs.",
    "Write a response to a negative code review comment that disagrees with the reviewer.",
    "I need to document a non-obvious design decision so future engineers understand it.",
    "Help me write a one-liner that explains what attention mechanisms do.",
    "I need to estimate how long a project will take and I have no idea. How should I think about it?",
    "Write a short slack message asking the team to review a PR by end of day.",
    "Help me write a question to ask at the end of a job interview.",
    "I need to present bad news (the model didn't improve) to my team. How should I frame it?",
    "Write a brief description of what LoRA is for a model card on HuggingFace.",
]


# ── Model loading ─────────────────────────────────────────────────────────────

def get_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_model(adapter_path: str | None):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=get_bnb_config(),
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"  Loaded adapter: {adapter_path}")
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── Eval 1: Conversational Quality ───────────────────────────────────────────

def judge_quality(prompt: str, response: str) -> float:
    """Ask Claude Haiku to score conversational quality 1–5."""
    client = anthropic.Anthropic()
    judge_prompt = f"""Rate this AI response on conversational naturalness and appropriate conciseness.

User message: {prompt}
Response: {response}

Score on a 1–5 scale:
5 = Feels like a smart, direct person texting back. No filler. Gets to the point. Warm but efficient.
4 = Mostly natural, maybe one unnecessary sentence or phrase.
3 = Serviceable but noticeably formal or padded.
2 = Over-explained or filled with bureaucratic warmup ("Certainly!", "Great question!").
1 = Exhausting to read. Restates the question, adds unnecessary structure, heavily padded.

Reply with ONLY a number from 1 to 5. No explanation."""

    response_obj = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=5,
        messages=[{"role": "user", "content": judge_prompt}],
    )
    try:
        return float(response_obj.content[0].text.strip())
    except ValueError:
        return 3.0  # neutral fallback if judge returns unexpected output


def run_quality_eval(model, tokenizer, n: int = 50) -> dict:
    prompts = QUALITY_EVAL_PROMPTS[:n]
    scores = []
    results = []

    for i, prompt in enumerate(prompts):
        print(f"  [{i+1}/{n}] {prompt[:60]}...")
        response = generate(model, tokenizer, prompt)
        score = judge_quality(prompt, response)
        scores.append(score)
        results.append({"prompt": prompt, "response": response, "score": score})
        print(f"    score: {score}/5")
        time.sleep(0.3)

    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    std = variance ** 0.5
    return {"mean": mean, "std": std, "n": n, "scores": scores, "results": results}


# ── Eval 2: TruthfulQA ───────────────────────────────────────────────────────

def run_truthfulqa_eval(model, tokenizer, n: int = 100) -> dict:
    dataset = hf_load_dataset("truthful_qa", "multiple_choice", split="validation")
    dataset = dataset.shuffle(seed=42).select(range(n))

    correct = 0
    results = []

    for i, row in enumerate(dataset):
        question = row["question"]
        choices = row["mc1_targets"]["choices"]
        labels = row["mc1_targets"]["labels"]
        correct_idx = labels.index(1)

        options_text = "\n".join(f"{chr(65+j)}. {c}" for j, c in enumerate(choices))
        prompt = (
            f"Question: {question}\n\nOptions:\n{options_text}\n\n"
            "Answer with only the letter of the correct option."
        )

        response = generate(model, tokenizer, prompt, max_new_tokens=10)
        predicted = response.strip()[0].upper() if response.strip() else "?"
        expected = chr(65 + correct_idx)
        is_correct = predicted == expected
        if is_correct:
            correct += 1

        results.append({
            "question": question,
            "correct_answer": choices[correct_idx],
            "predicted": predicted,
            "expected": expected,
            "correct": is_correct,
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n}] running accuracy: {correct/(i+1):.1%}")

    accuracy = correct / n
    return {"accuracy": accuracy, "correct": correct, "n": n, "results": results}


# ── Eval 3: Helpfulness ──────────────────────────────────────────────────────

def judge_helpfulness(prompt: str, response: str) -> bool:
    """Ask Claude Haiku whether the response was actually helpful. Returns True if helpful."""
    client = anthropic.Anthropic()
    judge_prompt = f"""A user sent a task request to an AI assistant. Was the response genuinely helpful?

User request: {prompt}
Response: {response}

Helpful means: the response actually addressed the request in a usable way. A response that is too brief to be useful, too vague to act on, or that refuses without reason is NOT helpful.

Reply with ONLY "yes" or "no"."""

    response_obj = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=5,
        messages=[{"role": "user", "content": judge_prompt}],
    )
    verdict = response_obj.content[0].text.strip().lower()
    return verdict.startswith("yes")


def run_helpfulness_eval(model, tokenizer, n: int = 50) -> dict:
    prompts = HELPFULNESS_EVAL_PROMPTS[:n]
    helpful_count = 0
    results = []

    for i, prompt in enumerate(prompts):
        print(f"  [{i+1}/{n}] {prompt[:60]}...")
        response = generate(model, tokenizer, prompt, max_new_tokens=400)
        helpful = judge_helpfulness(prompt, response)
        if helpful:
            helpful_count += 1
        results.append({"prompt": prompt, "response": response, "helpful": helpful})
        print(f"    {'helpful' if helpful else 'NOT helpful'} ({helpful_count}/{i+1})")
        time.sleep(0.3)

    helpfulness_rate = helpful_count / n
    return {"helpfulness_rate": helpfulness_rate, "helpful_count": helpful_count, "n": n, "results": results}


# ── Orchestration ─────────────────────────────────────────────────────────────

def eval_checkpoint(
    name: str,
    adapter_path: str | None,
    eval_type: str,
    n_quality: int,
    n_truthfulqa: int,
    n_helpfulness: int,
    output_dir: str,
) -> dict:
    """Load one checkpoint, run the requested evals, save results to output_dir/name.json."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {name} (adapter: {adapter_path or 'none — base model'})")
    print(f"{'='*60}")

    model, tokenizer = load_model(adapter_path)
    report = {"name": name, "adapter": adapter_path}

    if eval_type in ("quality", "all"):
        print(f"\n── Conversational Quality ({n_quality} prompts) ──")
        report["quality"] = run_quality_eval(model, tokenizer, n=n_quality)
        print(f"  Mean score: {report['quality']['mean']:.2f} ± {report['quality']['std']:.2f}")

    if eval_type in ("truthfulqa", "all"):
        print(f"\n── TruthfulQA ({n_truthfulqa} questions) ──")
        report["truthfulqa"] = run_truthfulqa_eval(model, tokenizer, n=n_truthfulqa)
        print(f"  Accuracy: {report['truthfulqa']['accuracy']:.1%}")

    if eval_type in ("helpfulness", "all"):
        print(f"\n── Helpfulness ({n_helpfulness} prompts) ──")
        report["helpfulness"] = run_helpfulness_eval(model, tokenizer, n=n_helpfulness)
        print(f"  Helpfulness rate: {report['helpfulness']['helpfulness_rate']:.1%}")

    # Save per-checkpoint results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / f"{name}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved to {out_path}")

    # Free GPU memory before next checkpoint
    del model
    torch.cuda.empty_cache()

    return report


def print_summary(reports: list[dict], eval_type: str):
    print("\n" + "═" * 70)
    print("SUMMARY")
    print("═" * 70)

    if eval_type in ("quality", "all"):
        print(f"\n{'Checkpoint':<18} {'Quality (mean ± std)':>22}")
        print("-" * 42)
        for r in reports:
            if "quality" in r:
                q = r["quality"]
                print(f"{r['name']:<18} {q['mean']:>8.2f} ± {q['std']:.2f}")

    if eval_type in ("truthfulqa", "all"):
        print(f"\n{'Checkpoint':<18} {'TruthfulQA Accuracy':>22}")
        print("-" * 42)
        for r in reports:
            if "truthfulqa" in r:
                print(f"{r['name']:<18} {r['truthfulqa']['accuracy']:>21.1%}")

    if eval_type in ("helpfulness", "all"):
        print(f"\n{'Checkpoint':<18} {'Helpfulness Rate':>22}")
        print("-" * 42)
        for r in reports:
            if "helpfulness" in r:
                print(f"{r['name']:<18} {r['helpfulness']['helpfulness_rate']:>21.1%}")

    print()


def main(
    adapters: list[tuple[str, str | None]],
    eval_type: str,
    n_quality: int,
    n_truthfulqa: int,
    n_helpfulness: int,
    output_dir: str,
):
    reports = []
    for name, adapter_path in adapters:
        report = eval_checkpoint(
            name=name,
            adapter_path=adapter_path,
            eval_type=eval_type,
            n_quality=n_quality,
            n_truthfulqa=n_truthfulqa,
            n_helpfulness=n_helpfulness,
            output_dir=output_dir,
        )
        reports.append(report)

    # Save combined summary
    summary_path = Path(output_dir) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(reports, f, indent=2)
    print(f"Full results saved to {summary_path}")

    print_summary(reports, eval_type)


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set.")
        raise SystemExit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter", action="append", dest="adapters", default=[],
        metavar="NAME:PATH",
        help="Named adapter to evaluate, e.g. sft:outputs/sft-run/final (repeatable)",
    )
    parser.add_argument(
        "--base", action="store_true",
        help="Always evaluate the base model (no adapter)",
    )
    parser.add_argument(
        "--eval", choices=["quality", "truthfulqa", "helpfulness", "all"], default="all",
    )
    parser.add_argument("--n-quality", type=int, default=50)
    parser.add_argument("--n-truthfulqa", type=int, default=100)
    parser.add_argument("--n-helpfulness", type=int, default=50)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    # Build ordered checkpoint list: base first, then named adapters
    checkpoints = []
    if args.base or not args.adapters:
        checkpoints.append(("base", None))
    for entry in args.adapters:
        if ":" not in entry:
            parser.error(f"--adapter must be NAME:PATH, got: {entry!r}")
        name, path = entry.split(":", 1)
        checkpoints.append((name, path))

    main(
        adapters=checkpoints,
        eval_type=args.eval,
        n_quality=args.n_quality,
        n_truthfulqa=args.n_truthfulqa,
        n_helpfulness=args.n_helpfulness,
        output_dir=args.output_dir,
    )
