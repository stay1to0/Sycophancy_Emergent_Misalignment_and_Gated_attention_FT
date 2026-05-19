import json
import os
import sys

from datasets import Dataset
from unsloth import FastLanguageModel

from validate import TrainingConfig
from sft import sft_train
from utils import load_jsonl, load_model_and_tokenizer
from config import setup_credentials


# Set up credentials and environment (HF token / etc.)
setup_credentials()


def _load_rows(training_file):
    """training_file can be str or list[str]."""
    if isinstance(training_file, list):
        rows = []
        for fp in training_file:
            rows.extend(load_jsonl(fp))
        return rows
    return load_jsonl(training_file)


def _build_sft_dataset(rows):
    """
    Expect each row to have: {"messages": [...]}
    Keep only messages to avoid carrying extra keys into Dataset.
    """
    return Dataset.from_list([{"messages": r["messages"]} for r in rows])


def train(training_cfg: TrainingConfig):
    # Guard: this script is now SFT-only by design
    if training_cfg.loss != "sft":
        raise ValueError(f"This training.py is SFT-only. Got loss={training_cfg.loss}")

    # Load model & tokenizer
    model, tokenizer = load_model_and_tokenizer(
        training_cfg.model,
        load_in_4bit=training_cfg.load_in_4bit,
    )

    # Build (or skip) LoRA adapter
    if training_cfg.is_peft:
        target_modules = training_cfg.target_modules or []
        print("Creating new LoRA adapter with target_modules =", target_modules)
        model = FastLanguageModel.get_peft_model(
            model,
            r=training_cfg.r,
            target_modules=target_modules,
            lora_alpha=training_cfg.lora_alpha,
            lora_dropout=training_cfg.lora_dropout,
            bias=training_cfg.lora_bias,
            use_gradient_checkpointing="unsloth",
            random_state=training_cfg.seed,
            use_rslora=training_cfg.use_rslora,
            loftq_config=None,
            use_dora=False,
        )
    else:
        print("PEFT disabled (is_peft=false). Training full model parameters.")

    # Load training data
    rows = _load_rows(training_cfg.training_file)
    dataset = _build_sft_dataset(rows)

    # Output dir & config dump
    os.makedirs(training_cfg.output_dir, exist_ok=True)
    with open(os.path.join(training_cfg.output_dir, "training_config.json"), "w") as f:
        json.dump(training_cfg.model_dump(), f, ensure_ascii=False, indent=2)

    # Prepare test dataset
    if training_cfg.test_file:
        test_rows = load_jsonl(training_cfg.test_file)
        test_dataset = _build_sft_dataset(test_rows)
    else:
        if training_cfg.no_test_split:
            # No test split requested; reuse train set as eval set (still required by trainer)
            test_dataset = dataset
        else:
            split = dataset.train_test_split(test_size=0.1, seed=training_cfg.seed)
            dataset = split["train"]
            test_dataset = split["test"]

    # Optional kwargs
    kwargs = {}
    if training_cfg.max_steps:
        kwargs["max_steps"] = training_cfg.max_steps

    trainer = sft_train(
        training_cfg=training_cfg,
        dataset=dataset,
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        **kwargs,
    )
    trainer.train()
    print("Training complete. Model saved to", training_cfg.output_dir)


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python training.py <path_to_config.json>")

    with open(sys.argv[1], "r") as f:
        config = json.load(f)

    training_config = TrainingConfig(**config)
    train(training_config)


if __name__ == "__main__":
    main()
