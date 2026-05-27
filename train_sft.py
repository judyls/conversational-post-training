"""
Conversational Post-Training — SFT Phase
Fine-tunes Qwen 2.5 7B on chosen (conversational) responses only.
Saves adapter before DPO — compare SFT-only vs SFT+DPO in evals.

Usage (on A10G):
    python train_sft.py
    python train_sft.py --output-dir outputs/sft-run --epochs 2
"""

import json
import argparse

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DATA_FILE = "data/preference_pairs.json"

SYSTEM_PROMPT = (
    "You are a helpful assistant. Be direct, concise, and natural. "
    "Get to the point without filler or padding."
)


def load_dataset(path: str, tokenizer) -> Dataset:
    with open(path) as f:
        records = json.load(f)

    def format_example(record):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": record["prompt"]},
            {"role": "assistant", "content": record["chosen"]},
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

    dataset = Dataset.from_list(records)
    return dataset.map(format_example, remove_columns=dataset.column_names)


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


def main(output_dir: str, epochs: int, batch_size: int):
    print(f"Loading {BASE_MODEL} in 4-bit...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.padding_side = "right"  # SFT uses right-padding (unlike DPO which uses left)

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
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, get_lora_config())
    model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_seq_length=512,
        dataset_text_field="text",
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

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
    )

    print("Starting SFT training...")
    trainer.train()

    print(f"Saving SFT adapter to {output_dir}/final/")
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    print("Done. Run train_dpo.py next, pointing --sft-adapter here.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/sft-run")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    main(output_dir=args.output_dir, epochs=args.epochs, batch_size=args.batch_size)
