"""
Conversational Post-Training — Modal SFT Runner

Setup (one-time):
    modal secret create huggingface-secret HF_TOKEN=hf_...
    modal volume create conversational-post-training-outputs

Run:
    modal run modal_train_sft.py

Download adapter when done:
    modal volume get conversational-post-training-outputs /sft-run ./outputs/sft-run
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
        "sentencepiece",
        "rich",
    )
)

volume = modal.Volume.from_name("conversational-post-training-outputs", create_if_missing=True)

app = modal.App("conversational-post-training-sft")

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
def run_sft(epochs: int = 2, batch_size: int = 4):
    import sys
    import os

    _checkout_pinned("/app")
    sys.path.insert(0, "/app")
    os.chdir("/app")

    # Copy data from volume if it exists there, otherwise fail loudly
    data_src = "/outputs/data/preference_pairs.json"
    data_dst = "/app/data/preference_pairs.json"
    import shutil
    import pathlib
    pathlib.Path("/app/data").mkdir(exist_ok=True)
    if pathlib.Path(data_src).exists():
        shutil.copy(data_src, data_dst)
        print(f"Loaded preference_pairs.json from volume ({pathlib.Path(data_src).stat().st_size // 1024}KB)")
    else:
        raise FileNotFoundError(
            f"{data_src} not found on volume. "
            "Upload first: modal volume put conversational-post-training-outputs data/preference_pairs.json /data/preference_pairs.json"
        )

    from train_sft import main
    main(output_dir="/outputs/sft-run", epochs=epochs, batch_size=batch_size)

    volume.commit()
    print("\nSFT adapter saved to Modal volume: conversational-post-training-outputs/sft-run")
    print("Download: modal volume get conversational-post-training-outputs /sft-run ./outputs/sft-run")
    print("Next: modal run modal_train_dpo.py --beta 0.1")


@app.local_entrypoint()
def main(epochs: int = 2, batch_size: int = 4):
    run_sft.remote(epochs=epochs, batch_size=batch_size)
