#!/usr/bin/env python
# coding: utf-8

import os
import sys
from pathlib import Path
import argparse

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer

# ---------------------------------------------------------
# ✅ Ensure we import LOCAL modeling_qwen3.py / configuration_qwen3.py
# ---------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from configuration_qwen3 import Qwen3Config
from modeling_qwen3 import Qwen3ForCausalLM


def main():
    parser = argparse.ArgumentParser(
        description="SFT: train ONLY gate_proj_myself, save gating_params.pth (local modeling_qwen3.py)"
    )
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./gated_qwen3_sft_myself")
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--attn_impl", type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) tokenizer
    print(f"[1] Loading tokenizer from: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    # 2) load config with LOCAL Qwen3Config
    print(f"[2] Loading config (local Qwen3Config) from: {args.model_name_or_path}")
    config = Qwen3Config.from_pretrained(args.model_name_or_path)

    # ✅ force disable OFFICIAL gates (your requirement)
    if hasattr(config, "headwise_attn_output_gate"):
        config.headwise_attn_output_gate = False
    if hasattr(config, "elementwise_attn_output_gate"):
        config.elementwise_attn_output_gate = False

    # ✅ force attention implementation for stability
    # Qwen3DecoderLayer uses config._attn_implementation at init time
    if hasattr(config, "_attn_implementation"):
        config._attn_implementation = args.attn_impl

    # 3) load model with LOCAL Qwen3ForCausalLM
    print(f"[3] Loading model weights (local Qwen3ForCausalLM) from: {args.model_name_or_path}")
    model = Qwen3ForCausalLM.from_pretrained(args.model_name_or_path, config=config)

    # recommended: disable cache during training
    model.config.use_cache = False

    # 4) freeze all params; only train gate_proj_myself
    print("[4] Freezing all params; unfreezing gate_proj_myself only")
    for p in model.parameters():
        p.requires_grad = False

    # re-init ALL gate_proj_myself to (weight=0, bias=10) to match your desired init
    gate_found = 0
    for module in model.modules():
        if hasattr(module, "gate_proj_myself"):
            gp = module.gate_proj_myself
            with torch.no_grad():
                torch.nn.init.zeros_(gp.weight)
                if gp.bias is not None:
                    torch.nn.init.constant_(gp.bias, 0.0)
            gate_found += 1

    if gate_found == 0:
        raise RuntimeError(
            "No gate_proj_myself found in model. "
            "Please confirm you have inserted gate_proj_myself into Qwen3Attention in your local modeling_qwen3.py."
        )
    print(f"  - Found gate_proj_myself in {gate_found} attention modules")

    for name, p in model.named_parameters():
        if "gate_proj_myself" in name:
            p.requires_grad = True

    # sanity check
    print("[4.1] Trainable-parameter sanity check:")
    unexpected = []
    for name, p in model.named_parameters():
        if p.requires_grad and "gate_proj_myself" not in name:
            unexpected.append(name)
    if unexpected:
        print("⚠️ Unexpected trainable params:")
        for n in unexpected:
            print("   ", n)
        raise RuntimeError("Unexpected trainable params detected. Aborting to stay strict.")
    else:
        print("  ✅ Only gate_proj_myself is trainable.")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[4.2] Total params: {total_params}")
    print(f"[4.2] Trainable params: {trainable_params} ({trainable_params/total_params*100:.6f}%)")

    # 5) dataset
    print(f"[5] Loading dataset from: {args.train_file}")
    raw_dataset = load_dataset("json", data_files={"train": args.train_file})
    train_data = raw_dataset["train"]
    max_seq_len = args.max_seq_len

    def preprocess(example):
        messages = example.get("messages", None)
        if messages is None:
            raise ValueError("Each example must contain a `messages` field.")
        if messages[-1]["role"] != "assistant":
            raise ValueError("The last message must be from role='assistant' for SFT supervision.")

        prefix_messages = messages[:-1]

        prompt_text = tokenizer.apply_chat_template(
            prefix_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )

        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)

        # ensure eos
        if tokenizer.eos_token_id is not None and (len(full_ids) == 0 or full_ids[-1] != tokenizer.eos_token_id):
            full_ids.append(tokenizer.eos_token_id)

        # labels: only supervise assistant output
        if len(full_ids) < len(prompt_ids) or full_ids[:len(prompt_ids)] != prompt_ids:
            # fallback (robustness)
            full_ids = tokenizer.encode(prompt_text + messages[-1]["content"], add_special_tokens=False)
            if tokenizer.eos_token_id is not None and (len(full_ids) == 0 or full_ids[-1] != tokenizer.eos_token_id):
                full_ids.append(tokenizer.eos_token_id)
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

        input_ids = full_ids
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

        # left truncate to keep tail
        if len(input_ids) > max_seq_len:
            overflow = len(input_ids) - max_seq_len
            input_ids = input_ids[overflow:]
            labels = labels[overflow:]

        return {"input_ids": input_ids, "labels": labels}

    train_data = train_data.map(
        preprocess,
        remove_columns=train_data.column_names,
        desc="Tokenizing & building labels",
    )

    def data_collator(features):
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids, labels, attention_mask = [], [], []
        for f in features:
            ids = f["input_ids"]
            labs = f["labels"]
            pad = max_len - len(ids)

            input_ids.append(ids + [tokenizer.pad_token_id] * pad)
            labels.append(labs + [-100] * pad)
            attention_mask.append([1] * len(ids) + [0] * pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    # 6) training args
    print("[6] Building TrainingArguments")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=1,
        prediction_loss_only=True,
        save_strategy="no",
        report_to="none",
        fp16=args.fp16,
        bf16=args.bf16,
    )

    # 7) trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        data_collator=data_collator,
    )


    # ============================================================
    # 🔎 DEBUG STEP: single-batch gradient sanity check
    # ============================================================
    print("\n[DEBUG] Running single-step gradient check for gate_proj_myself...\n")

    device = model.device
    model.train()

    dl = trainer.get_train_dataloader()
    batch = next(iter(dl))
    batch = {k: v.to(device) for k, v in batch.items()}

    out = model(**batch)
    loss = out.loss
    print("[DEBUG] loss =", float(loss))

    loss.backward()

    gate_has_grad = False
    for name, p in model.named_parameters():
        if "gate_proj_myself" in name:
            if p.grad is None:
                print(f"[DEBUG] {name}: grad=None ❌")
            else:
                g = p.grad.detach()
                print(
                    f"[DEBUG] {name}: "
                    f"grad_abs_max={float(g.abs().max()):.3e}, "
                    f"grad_abs_mean={float(g.abs().mean()):.3e}"
                )
                gate_has_grad = True

    if not gate_has_grad:
        print("[DEBUG] ❌ No gate_proj_myself parameters received gradients!")
    else:
        print("[DEBUG] ✅ gate_proj_myself gradient check completed.")

    # ⚠️ 重要：防止后续 trainer.train() 累积旧梯度
    model.zero_grad(set_to_none=True)

    print("\n[DEBUG] Gradient sanity check finished.\n")
    # ============================================================


    print("[7] Starting training...")
    trainer.train()
    print("[7] Training completed.")

    # 8) save gate params only
    print("[8] Saving gating_params.pth ...")
    gate_state = {k: v.cpu() for k, v in model.state_dict().items() if "gate_proj_myself" in k}
    torch.save(gate_state, os.path.join(args.output_dir, "gating_params.pth"))
    print(f"[8] Saved to: {args.output_dir}/gating_params.pth")


if __name__ == "__main__":
    main()
