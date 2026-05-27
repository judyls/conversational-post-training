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
        "trl==0.9.6",
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


@app.function(
    gpu="A10G",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/outputs": volume},
    timeout=60 * 60 * 3,
)
def run_dpo(beta: float = 0.1, epochs: int = 3, batch_size: int = 2):
    import subprocess
    import sys
    import os
    import shutil
    import pathlib

    subprocess.run(["git", "clone", GITHUB_REPO, "/app"], check=True)
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
