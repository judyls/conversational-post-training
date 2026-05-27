"""
Conversational Post-Training — Dataset Generation
Generates preference pairs: chosen = conversational/brief/natural, rejected = verbose/padded/formal.

Three prompt categories:
  - factual    (30%): questions with unambiguous answers
  - conversational (40%): casual questions or dialogue starters where pacing matters
  - task       (30%): "help me write X", "explain Y", "what should I do about Z"

Usage:
    export ANTHROPIC_API_KEY=your_key
    python generate_dataset.py              # full run (~400 pairs)
    python generate_dataset.py --test       # test run (9 pairs, 3 per category)
    python generate_dataset.py --n 200      # smaller run
"""

import anthropic
import json
import time
import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env.local")

client = anthropic.Anthropic()

PROMPT_MODEL = "claude-haiku-4-5-20251001"  # cheap — for generating the prompts
PAIR_MODEL = "claude-sonnet-4-6"            # quality — for generating chosen/rejected pairs

CHECKPOINT_FILE = "data/checkpoint.json"
PROMPTS_CACHE_FILE = "data/prompts_cache.json"
OUTPUT_FILE = "data/preference_pairs.json"

CATEGORIES = {
    "factual": {
        "ratio": 0.30,
        "description": (
            "factual questions with unambiguous correct answers "
            "(science, history, math, geography, technology)"
        ),
        "prompt_instruction": (
            "Each should be a clear question with a definite correct answer, "
            "the kind someone might ask an AI assistant in a casual conversation."
        ),
    },
    "conversational": {
        "ratio": 0.40,
        "description": (
            "casual conversational questions or dialogue starters — "
            "opinions, recommendations, personal advice, or open-ended questions "
            "where natural pacing and tone matter most"
        ),
        "prompt_instruction": (
            "Examples: 'Any good podcasts for commuting?', "
            "'Should I learn Rust or Go first?', "
            "'My flight was delayed 4 hours, what do I do?' "
            "Mix of everyday decisions, casual curiosity, and light advice-seeking."
        ),
    },
    "task": {
        "ratio": 0.30,
        "description": (
            "task requests — 'help me write X', 'explain Y to me', "
            "'review this', 'debug this', 'summarize this'"
        ),
        "prompt_instruction": (
            "Examples: 'Help me write a short bio for LinkedIn', "
            "'Explain recursion like I'm a junior dev', "
            "'What's wrong with this regex: [0-9+]?' "
            "Should feel like realistic requests to an AI assistant."
        ),
    },
}


def call_api(prompt: str, model: str, max_tokens: int = 2000, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except anthropic.RateLimitError:
            wait = 2 ** attempt * 5
            print(f"  Rate limited, waiting {wait}s...")
            time.sleep(wait)
        except anthropic.APIError as e:
            print(f"  API error (attempt {attempt+1}): {e}")
            time.sleep(2)
    raise RuntimeError(f"Failed after {retries} attempts")


def parse_json_response(text: str) -> any:
    """Extract JSON from response, handling markdown code fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text)


def generate_prompts(category: str, n: int) -> list[str]:
    """Generate n prompts for the given category using Haiku."""
    cat = CATEGORIES[category]
    print(f"  Generating {n} {category} prompts...")

    prompt = f"""Generate {n} prompts in this category: {cat['description']}.

{cat['prompt_instruction']}

Rules:
- Each prompt should be 1-3 sentences max, the way a real person would type it
- Vary the phrasing — don't repeat the same sentence structure
- No prompt should require context ("as I mentioned earlier", "following up on...")
- Each should stand alone

Return ONLY a JSON array of strings, no other text:
["...", "...", ...]"""

    text = call_api(prompt, PROMPT_MODEL, max_tokens=4000)
    prompts = parse_json_response(text)
    print(f"    Got {len(prompts)} prompts")
    return prompts


def generate_preference_pair(prompt: str, category: str) -> dict:
    """Generate chosen (conversational) and rejected (verbose) responses using Sonnet."""
    cat = CATEGORIES[category]

    api_prompt = f"""A user sent this message to an AI assistant:

"{prompt}"

Category: {cat['description']}

Generate two responses:

1. CHOSEN — conversational, brief, natural. Feels like a thoughtful person texting back.
   Rules: No opening filler ("Great question!", "Certainly!", "Of course!"). No unnecessary hedging.
   No padding. Bullet points only if genuinely the clearest format. Gets to the point immediately.
   Warm but efficient. 1-4 sentences for simple things; longer only when the task genuinely requires it.

2. REJECTED — verbose, over-explained, formal. The kind of response that exhausts you to read.
   Rules: Must start with filler or preamble. Should over-explain obvious things. Add unnecessary
   structure (headers, bullets for simple answers). Use formal language where casual works better.
   Pad with caveats and disclaimers. At least 2x longer than the chosen response.

Return ONLY JSON, no other text:
{{"chosen": "...", "rejected": "..."}}"""

    text = call_api(api_prompt, PAIR_MODEL, max_tokens=1000)
    return parse_json_response(text)


def load_checkpoint() -> dict:
    if Path(CHECKPOINT_FILE).exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"completed_indices": [], "dataset": []}


def save_checkpoint(state: dict):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(state, f)


def build_prompts_cache(n_total: int) -> list[dict]:
    """Generate or load all prompts, tagged by category."""
    if Path(PROMPTS_CACHE_FILE).exists():
        with open(PROMPTS_CACHE_FILE) as f:
            cached = json.load(f)
        print(f"Loaded {len(cached)} prompts from cache")
        return cached

    print(f"Generating {n_total} prompts across {len(CATEGORIES)} categories...")
    all_prompts = []

    for category, cat in CATEGORIES.items():
        n = round(n_total * cat["ratio"])
        prompts = generate_prompts(category, n)
        for p in prompts:
            all_prompts.append({"prompt": p, "category": category})
        time.sleep(0.5)

    with open(PROMPTS_CACHE_FILE, "w") as f:
        json.dump(all_prompts, f, indent=2)
    print(f"Cached {len(all_prompts)} prompts to {PROMPTS_CACHE_FILE}")
    return all_prompts


def build_dataset(n_total: int = 400) -> list[dict]:
    Path("data").mkdir(exist_ok=True)
    state = load_checkpoint()
    dataset = state["dataset"]
    completed = set(state["completed_indices"])

    print(f"Checkpoint: {len(completed)} prompts done, {len(dataset)} pairs in dataset")

    all_prompts = build_prompts_cache(n_total)
    total = len(all_prompts)

    for i, entry in enumerate(all_prompts):
        if i in completed:
            continue

        prompt = entry["prompt"]
        category = entry["category"]
        print(f"\n[{i+1}/{total}] [{category}] {prompt[:70]}...")

        try:
            pair = generate_preference_pair(prompt, category)
            dataset.append({
                "prompt": prompt,
                "category": category,
                "chosen": pair["chosen"],
                "rejected": pair["rejected"],
            })
            print(f"  chosen:   {pair['chosen'][:100]}...")
            print(f"  rejected: {pair['rejected'][:100]}...")

        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Parse error, skipping: {e}")
            completed.add(i)
            state["completed_indices"] = list(completed)
            state["dataset"] = dataset
            save_checkpoint(state)
            time.sleep(0.5)
            continue
        except Exception as e:
            print(f"  Error: {e}, saving checkpoint and exiting")
            state["completed_indices"] = list(completed)
            state["dataset"] = dataset
            save_checkpoint(state)
            sys.exit(1)

        completed.add(i)
        state["completed_indices"] = list(completed)
        state["dataset"] = dataset
        save_checkpoint(state)
        time.sleep(0.5)

    return dataset


def run_test():
    """Generate 9 pairs (3 per category) and print them for quality review."""
    print("=== TEST RUN — 3 prompts per category = 9 pairs ===\n")
    Path("data").mkdir(exist_ok=True)

    results = []
    for category in CATEGORIES:
        prompts = generate_prompts(category, 3)
        print(f"\n── {category.upper()} ──")
        for prompt in prompts:
            print(f"\nPrompt: {prompt}")
            try:
                pair = generate_preference_pair(prompt, category)
                print(f"CHOSEN:   {pair['chosen']}")
                print(f"REJECTED: {pair['rejected'][:200]}...")
                results.append({
                    "prompt": prompt,
                    "category": category,
                    "chosen": pair["chosen"],
                    "rejected": pair["rejected"],
                })
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  Parse error: {e}")
            time.sleep(0.5)

    with open("data/test_pairs.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nSaved {len(results)} test pairs to data/test_pairs.json")
    print("Review them — if quality looks good, run without --test for the full dataset.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run a small test batch first")
    parser.add_argument("--n", type=int, default=400, help="Total number of pairs to generate (default: 400)")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set.")
        print("Run: export ANTHROPIC_API_KEY=your_key")
        sys.exit(1)

    if args.test:
        run_test()
    else:
        dataset = build_dataset(n_total=args.n)

        with open(OUTPUT_FILE, "w") as f:
            json.dump(dataset, f, indent=2)

        print(f"\n✓ Done. {len(dataset)} preference pairs saved to {OUTPUT_FILE}")
        category_counts = {}
        for entry in dataset:
            category_counts[entry["category"]] = category_counts.get(entry["category"], 0) + 1
        for cat, count in sorted(category_counts.items()):
            print(f"  {cat}: {count}")
        print("Next: review a sample, then run train_sft.py")
