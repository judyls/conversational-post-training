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
    import subprocess
    import sys
    import os

    subprocess.run(["git", "clone", GITHUB_REPO, "/app"], check=True)
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
