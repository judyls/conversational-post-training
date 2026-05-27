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


@app.function(
    gpu="A10G",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/outputs": volume},
    timeout=60 * 60 * 3,
)
def run_sft(epochs: int = 2, batch_size: int = 4):
    import subprocess
    import sys
    import os

    subprocess.run(["git", "clone", GITHUB_REPO, "/app"], check=True)
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
