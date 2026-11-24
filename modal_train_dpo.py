"""
Conversational Post-Training — Modal DPO Runner
Run three times with different beta values — that's the research.

Setup: SFT must have run first (modal_train_sft.py).

Run all three beta experiments:
    modal run modal_train_dpo.py --beta 0.05
    modal run modal_train_dpo.py --beta 0.1
    modal run modal_train_dpo.py --beta 0.2

Download adapters:
    modal volume get conversational-post-training-outputs /dpo-beta005 ./outputs/dpo-beta005
    modal volume get conversational-post-training-outputs /dpo-beta010 ./outputs/dpo-beta010
    modal volume get conversational-post-training-outputs /dpo-beta020 ./outputs/dpo-beta020
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.5.1", extra_index_url="https://download.pytorch.org/whl/cu124")
    .pip_install(
        "transformers==4.44.2",
        "trl==0.10.1",
        "peft==0.12.0",
        "bitsandbytes==0.43.1",
        "accelerate==0.34.2",
        "datasets==2.20.0",
        "sentencepiece",
        "rich",
    )
)

volume = modal.Volume.from_name("conversational-post-training-outputs", create_if_missing=False)

app = modal.App("conversational-post-training-dpo")

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


@app.function(
    gpu="A10G",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/outputs": volume},
    timeout=60 * 60 * 3,
)
def run_dpo(beta: float = 0.1, epochs: int = 3, batch_size: int = 2):
    import sys
    import os
    import shutil
    import pathlib

    _checkout_pinned("/app")
    sys.path.insert(0, "/app")
    os.chdir("/app")

    # Copy data from volume
    pathlib.Path("/app/data").mkdir(exist_ok=True)
    shutil.copy("/outputs/data/preference_pairs.json", "/app/data/preference_pairs.json")

    # SFT adapter must exist on the volume
    sft_adapter_path = "/outputs/sft-run/final"
    if not pathlib.Path(sft_adapter_path).exists():
        raise FileNotFoundError(
            f"SFT adapter not found at {sft_adapter_path}. Run modal_train_sft.py first."
        )

    # Output dir named by beta: 0.05 → dpo-beta005, 0.1 → dpo-beta010, 0.2 → dpo-beta020
    beta_tag = f"{beta:.2f}".replace(".", "")
    output_dir = f"/outputs/dpo-beta{beta_tag}"

    from train_dpo import main
    main(
        sft_adapter=sft_adapter_path,
        output_dir=output_dir,
        beta=beta,
        epochs=epochs,
        batch_size=batch_size,
    )

    volume.commit()
    print(f"\nDPO adapter (beta={beta}) saved to Modal volume: conversational-post-training-outputs/dpo-beta{beta_tag}")
    print(f"Download: modal volume get conversational-post-training-outputs /dpo-beta{beta_tag} ./outputs/dpo-beta{beta_tag}")


@app.local_entrypoint()
def main(beta: float = 0.1, epochs: int = 3, batch_size: int = 2):
    run_dpo.remote(beta=beta, epochs=epochs, batch_size=batch_size)
