"""
Build the static data bundle for the comparison UI (docs/).

Reads the eval output downloaded from the Modal volume:
    modal volume get conversational-post-training-outputs /results ./results

Writes docs/data.json — every response shown in the UI is a real generation
from the eval run, not live inference and not hand-edited.

Run:
    python build_demo_data.py
"""

import json
import pathlib
from datetime import date

RESULTS = pathlib.Path("results")
PAGE = pathlib.Path("docs/index.html")

# Display order, labels and notes. base first (the "before"), then the trained runs.
# Labels are plain English because the page is written for readers who have never
# heard of SFT or DPO; the note carries the method's real name for those who have.
CHECKPOINTS = [
    ("base",    "Original",         None,  "The model as it ships, with no training from us"),
    ("sft",     "Imitate",          None,  "Shown 375 good answers to copy (SFT)"),
    ("dpo-005", "Compare · hard",   0.05, "Shown good and bad answers, pushed hardest (DPO, β=0.05)"),
    ("dpo-010", "Compare · medium", 0.10, "Shown good and bad answers, middle setting (DPO, β=0.10)"),
    ("dpo-020", "Compare · light",  0.20, "Shown good and bad answers, pushed least (DPO, β=0.20)"),
]


def load(name):
    return json.loads((RESULTS / f"{name}.json").read_text())


def words(text):
    return len(text.split())


def opener(text, n=3):
    """First n words, normalised — used to measure stock-phrase collapse."""
    return " ".join(text.split()[:n]).lower().strip(",.\u2014-")


def main():
    raw = {cid: load(cid) for cid, *_ in CHECKPOINTS}

    checkpoints = []
    for cid, label, beta, note in CHECKPOINTS:
        d = raw[cid]
        checkpoints.append({
            "id": cid,
            "label": label,
            "beta": beta,
            "note": note,
            "quality_mean": round(d["quality"]["mean"], 2),
            "quality_std": round(d["quality"]["std"], 2),
            "truthfulqa": round(d["truthfulqa"]["accuracy"] * 100, 1),
            "helpfulness": round(d["helpfulness"]["helpfulness_rate"] * 100, 1),
        })

    # Quality + helpfulness prompts are identical across checkpoints (verified),
    # so index by position off the base run.
    quality = []
    for i, r in enumerate(raw["base"]["quality"]["results"]):
        quality.append({
            "prompt": r["prompt"],
            "responses": {
                cid: {
                    "text": raw[cid]["quality"]["results"][i]["response"],
                    "score": raw[cid]["quality"]["results"][i]["score"],
                    "words": words(raw[cid]["quality"]["results"][i]["response"]),
                }
                for cid, *_ in CHECKPOINTS
            },
        })

    # Featured = the clearest base-vs-SFT contrasts: SFT scored at least as well
    # while cutting length most. These seed the UI for someone with no prompt
    # of their own in mind.
    def contrast(item):
        b, s_ = item["responses"]["base"], item["responses"]["sft"]
        if s_["score"] < b["score"] or b["words"] < 40:
            return -1
        return (b["words"] - s_["words"]) + 12 * (s_["score"] - b["score"])

    for item in sorted(quality, key=contrast, reverse=True)[:5]:
        item["featured"] = True

    helpfulness = []
    for i, r in enumerate(raw["base"]["helpfulness"]["results"]):
        helpfulness.append({
            "prompt": r["prompt"],
            "responses": {
                cid: {
                    "text": raw[cid]["helpfulness"]["results"][i]["response"],
                    "helpful": raw[cid]["helpfulness"]["results"][i]["helpful"],
                    "words": words(raw[cid]["helpfulness"]["results"][i]["response"]),
                }
                for cid, *_ in CHECKPOINTS
            },
        })

    # Featured helpfulness prompts = where SFT stayed helpful but the most
    # aggressive DPO run did not. This is the failure the brevity training
    # introduced, and the most useful thing to show someone.
    regressions = []
    for item in helpfulness:
        r = item["responses"]
        if r["base"]["helpful"] and r["sft"]["helpful"] and not r["dpo-005"]["helpful"]:
            item["dpo_regression"] = True
            regressions.append(item)

    # Only the few starkest get "start here" billing — biggest length collapse.
    regressions.sort(key=lambda it: it["responses"]["base"]["words"] - it["responses"]["dpo-005"]["words"], reverse=True)
    for item in regressions[:3]:
        item["featured"] = True

    # TruthfulQA stores only the chosen letter, not the option text, so the UI
    # shows the question, the reference answer, and who got it right.
    truthfulqa = []
    for i, r in enumerate(raw["base"]["truthfulqa"]["results"]):
        truthfulqa.append({
            "question": r["question"],
            "correct_answer": r["correct_answer"],
            "expected": r["expected"],
            "results": {
                cid: {
                    "predicted": raw[cid]["truthfulqa"]["results"][i]["predicted"],
                    "correct": raw[cid]["truthfulqa"]["results"][i]["correct"],
                }
                for cid, *_ in CHECKPOINTS
            },
        })

    # Style statistics computed across all 95 generations per checkpoint.
    # Mean length and opener concentration are a second, independent read on
    # collapse: a model repeating one stock opening has stopped responding to
    # the prompt. Not part of the original eval — derived here from its output.
    import collections
    for cp in checkpoints:
        cid = cp["id"]
        texts = ([it["responses"][cid]["text"] for it in quality]
                 + [it["responses"][cid]["text"] for it in helpfulness])
        counts = collections.Counter(opener(t) for t in texts)
        top = counts.most_common(3)
        cp["mean_words"] = round(sum(words(t) for t in texts) / len(texts), 1)
        cp["n_generations"] = len(texts)
        cp["top_openers"] = [
            {"phrase": phrase, "count": c, "share": round(c / len(texts) * 100, 1)}
            for phrase, c in top
        ]
        cp["opener_concentration"] = round(sum(c for _, c in top) / len(texts) * 100, 1)

    bundle = {
        "built": date.today().isoformat(),
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "checkpoints": checkpoints,
        "quality": quality,
        "helpfulness": helpfulness,
        "truthfulqa": truthfulqa,
    }

    # Embed the bundle in the page rather than writing it alongside. Browsers
    # block same-folder fetch() on file:// URLs, so a fetched bundle leaves the
    # page blank for anyone who just opens index.html from disk.
    blob = json.dumps(bundle, separators=(",", ":")).replace("<", "\\u003c")
    page = PAGE.read_text()
    start = page.index('<script id="bundle" type="application/json">') + len('<script id="bundle" type="application/json">')
    end = page.index("</script>", start)
    PAGE.write_text(page[:start] + blob + page[end:])
    kb = PAGE.stat().st_size / 1024
    print(f"wrote {PAGE} ({kb:.0f} KB, bundle embedded)")
    print(f"  {len(checkpoints)} checkpoints")
    print(f"  {len(quality)} quality prompts")
    print(f"  {len(helpfulness)} helpfulness prompts")
    print(f"  {len(truthfulqa)} TruthfulQA questions")


if __name__ == "__main__":
    main()
