"""
Conversational Post-Training — Modal Eval Runner
Evaluates all five checkpoints: base, SFT, DPO beta=0.05/0.1/0.2.

Setup: All three DPO runs must have completed first.

Run:
    modal run modal_eval.py                  # all checkpoints, all evals
    modal run modal_eval.py --eval quality   # conversational quality only

Download results:
    modal volume get conversational-post-training-outputs /results ./results
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.5.1", extra_index_url="https://download.pytorch.org/whl/cu124")
    .pip_install(
        "transformers==4.44.2",
        "trl==0.9.6",
        "peft==0.12.0",
        "bitsandbytes==0.43.1",
        "accelerate==0.34.2",
        "datasets==2.20.0",
        "anthropic>=0.25.0",
        "sentencepiece",
        "rich",
    )
)

volume = modal.Volume.from_name("conversational-post-training-outputs", create_if_missing=False)

app = modal.App("conversational-post-training-eval")

GITHUB_REPO = "https://github.com/judyls/conversational-post-training.git"

# The GPU container executes whatever this repo's code says, with the Hugging Face
# (and, for eval, Anthropic) secrets mounted. A bare `git clone` of the default branch
# means "whatever happens to be on main when this launches" — so the ref is pinned to an
# exact commit and the checkout is verified before any of it is imported. Bump GIT_REF
# deliberately when you want to run newer code.
GIT_REF = "0957beba0703ed720faa67415e9f84f1d671eb9d"


def _checkout_pinned(dest: str = "/app") -> None:
    """Fetch exactly GIT_REF into dest and refuse to continue if HEAD differs."""
    import subprocess

    if len(GIT_REF) != 40 or any(c not in "0123456789abcdef" for c in GIT_REF):
        raise RuntimeError(
            f"GIT_REF must be a full 40-character commit SHA, got {GIT_REF!r}. "
            "Branch and tag names are mutable and are not accepted here."
        )

    subprocess.run(["git", "init", "--quiet", dest], check=True)
    subprocess.run(["git", "-C", dest, "remote", "add", "origin", GITHUB_REPO], check=True)
    subprocess.run(["git", "-C", dest, "fetch", "--quiet", "--depth", "1", "origin", GIT_REF], check=True)
    subprocess.run(["git", "-C", dest, "checkout", "--quiet", "--detach", "FETCH_HEAD"], check=True)

    head = subprocess.run(
        ["git", "-C", dest, "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if head != GIT_REF:
        raise RuntimeError(f"refusing to run: pinned {GIT_REF}, checked out {head}")
    print(f"running pinned commit {head}")


# All five checkpoints to evaluate
CHECKPOINTS = [
    ("base",    None),
    ("sft",     "/outputs/sft-run/final"),
    ("dpo-005", "/outputs/dpo-beta005/final"),
    ("dpo-010", "/outputs/dpo-beta010/final"),
    ("dpo-020", "/outputs/dpo-beta020/final"),
]


@app.function(
    gpu="A10G",
    image=image,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("anthropic-secret"),
    ],
    volumes={"/outputs": volume},
    timeout=60 * 60 * 5,  # 5hr — evaluating 5 checkpoints takes longer than training
)
def run_eval(eval_type: str = "all", n_quality: int = 50, n_truthfulqa: int = 100, n_helpfulness: int = 50):
    import sys
    import os

    _checkout_pinned("/app")
    sys.path.insert(0, "/app")
    os.chdir("/app")

    # Verify all expected adapter paths exist before starting
    missing = []
    for name, path in CHECKPOINTS:
        if path and not __import__("pathlib").Path(path).exists():
            missing.append(f"{name}: {path}")
    if missing:
        print("WARNING: Some adapters not found on volume — skipping:")
        for m in missing:
            print(f"  {m}")

    available = [(name, path) for name, path in CHECKPOINTS if path is None or __import__("pathlib").Path(path).exists()]
    print(f"\nEvaluating {len(available)} checkpoints: {[n for n, _ in available]}")

    from eval import main
    main(
        adapters=available,
        eval_type=eval_type,
        n_quality=n_quality,
        n_truthfulqa=n_truthfulqa,
        n_helpfulness=n_helpfulness,
        output_dir="/outputs/results",
    )

    volume.commit()
    print("\nResults saved to Modal volume: conversational-post-training-outputs/results")
    print("Download: modal volume get conversational-post-training-outputs /results ./results")


@app.local_entrypoint()
def main(eval: str = "all", n_quality: int = 50, n_truthfulqa: int = 100, n_helpfulness: int = 50):
    run_eval.remote(
        eval_type=eval,
        n_quality=n_quality,
        n_truthfulqa=n_truthfulqa,
        n_helpfulness=n_helpfulness,
    )
