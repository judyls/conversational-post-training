"""
Conversational Post-Training — DPO Phase
Loads the SFT adapter as starting policy, runs DPO on preference pairs.

Three runs to compare (the research):
    python train_dpo.py --beta 0.05 --output-dir outputs/dpo-beta005
    python train_dpo.py --beta 0.1  --output-dir outputs/dpo-beta010
    python train_dpo.py --beta 0.2  --output-dir outputs/dpo-beta020

Usage:
    python train_dpo.py --sft-adapter outputs/sft-run/final --beta 0.1
"""

import json
import argparse

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DATA_FILE = "data/preference_pairs.json"

SYSTEM_PROMPT = (
    "You are a helpful assistant. Be direct, concise, and natural. "
    "Get to the point without filler or padding."
)


def load_dataset(path: str, tokenizer) -> Dataset:
    with open(path) as f:
        records = json.load(f)

    def format_prompt(record):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": record["prompt"]},
        ]
        return {
            "prompt": tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ),
            "chosen": record["chosen"],
            "rejected": record["rejected"],
        }

    dataset = Dataset.from_list(records)
    return dataset.map(format_prompt, remove_columns=["category"])


def get_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def get_lora_config() -> LoraConfig:
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


def main(sft_adapter: str | None, output_dir: str, beta: float, epochs: int, batch_size: int):
    print(f"Loading {BASE_MODEL} in 4-bit...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|im_end|>"
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    print(f"Loading dataset from {DATA_FILE}...")
    dataset = load_dataset(DATA_FILE, tokenizer)
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_data = split["train"]
    eval_data = split["test"]
    print(f"  Train: {len(train_data)}  Eval: {len(eval_data)}")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=get_bnb_config(),
        device_map={"": 0},
    )

    if sft_adapter:
        print(f"Loading SFT adapter from {sft_adapter}...")
        model = PeftModel.from_pretrained(model, sft_adapter, is_trainable=True)
    else:
        print("No SFT adapter — training DPO from base model (not recommended)")
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, get_lora_config())

    model.print_trainable_parameters()

    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        beta=beta,
        max_length=512,
        max_prompt_length=256,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
    )

    print(f"Starting DPO training (beta={beta})...")
    trainer.train()

    save_path = f"{output_dir}/final"
    print(f"Saving DPO adapter to {save_path}/")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Done. Run eval.py --adapter {save_path} next.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-adapter", default=None, help="Path to SFT adapter (recommended)")
    parser.add_argument("--output-dir", default="outputs/dpo-beta010")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    main(
        sft_adapter=args.sft_adapter,
        output_dir=args.output_dir,
        beta=args.beta,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
