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
MAX_LENGTH = 512
LABEL_PAD_ID = -100

SYSTEM_PROMPT = (
    "You are a helpful assistant. Be direct, concise, and natural. "
    "Get to the point without filler or padding."
)


def resolve_eos_token(tokenizer) -> tuple[str, int]:
    """Return (eos_token_str, eos_token_id) as plain str and int.

    Qwen 2.5's tokenizer config stores eos_token_id as a list and may leave
    eos_token as None. TRL 0.9.6 appends eos_token_id directly into integer
    lists, so it must be a plain int. We resolve it once here and use it
    explicitly everywhere rather than relying on tokenizer attributes.
    """
    # Try the tokenizer's own eos_token first
    tok_str = tokenizer.eos_token
    tok_id = tokenizer.eos_token_id

    # Unwrap list (Qwen 2.5 quirk)
    if isinstance(tok_id, list):
        tok_id = tok_id[0]

    # If eos_token is missing, derive from id
    if tok_str is None and tok_id is not None:
        tok_str = tokenizer.convert_ids_to_tokens(tok_id)

    # Final fallback for Qwen 2.5: <|im_end|> is the ChatML end token
    if tok_str is None:
        tok_str = "<|im_end|>"
        tok_id = tokenizer.convert_tokens_to_ids(tok_str)

    return tok_str, int(tok_id)


def load_dataset(path: str, tokenizer, eos_token: str, max_length: int) -> Dataset:
    """Pre-tokenize preference pairs into the exact columns DPOTrainer expects.

    TRL 0.9.6 skips its internal tokenize_row() when the dataset has no
    'chosen' column — so we produce the final token-ID columns ourselves,
    avoiding all issues with Qwen's non-standard eos_token_id.
    """
    with open(path) as f:
        records = json.load(f)

    def tokenize_example(record):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": record["prompt"]},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        n_prompt = len(prompt_ids)

        # Tokenize full sequences — prompt + response + eos — as one string
        # so the tokenizer handles any boundary effects consistently.
        chosen_ids = tokenizer(
            prompt_text + record["chosen"] + eos_token, add_special_tokens=False
        )["input_ids"][:max_length]

        rejected_ids = tokenizer(
            prompt_text + record["rejected"] + eos_token, add_special_tokens=False
        )["input_ids"][:max_length]

        # Labels: full sequence with prompt tokens masked so loss is only on response
        chosen_labels = [LABEL_PAD_ID] * n_prompt + chosen_ids[n_prompt:]
        rejected_labels = [LABEL_PAD_ID] * n_prompt + rejected_ids[n_prompt:]

        return {
            "prompt_input_ids":      prompt_ids,
            "prompt_attention_mask": [1] * len(prompt_ids),
            "chosen_input_ids":      chosen_ids,
            "chosen_attention_mask": [1] * len(chosen_ids),
            "chosen_labels":         chosen_labels,
            "rejected_input_ids":    rejected_ids,
            "rejected_attention_mask": [1] * len(rejected_ids),
            "rejected_labels":       rejected_labels,
        }

    dataset = Dataset.from_list(records)
    # remove_columns drops 'prompt', 'chosen', 'rejected', 'category' —
    # the absence of 'chosen' tells TRL 0.9.6 to skip its own tokenize_row()
    return dataset.map(tokenize_example, remove_columns=dataset.column_names)


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

    eos_token, eos_token_id = resolve_eos_token(tokenizer)
    # Ensure pad token is a plain integer (required by the data collator)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = eos_token
        tokenizer.pad_token_id = eos_token_id
    print(f"  eos_token={eos_token!r} ({eos_token_id}), pad_token={tokenizer.pad_token!r} ({tokenizer.pad_token_id})")

    print(f"Loading dataset from {DATA_FILE}...")
    dataset = load_dataset(DATA_FILE, tokenizer, eos_token=eos_token, max_length=MAX_LENGTH)
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
        max_length=MAX_LENGTH,
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
