#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clean-only training for LED-base clean.
RunPod/RTX 4090 safe version.

This script saves ONLY the final model:
    ./checkpoints/led_base_clean_runpod4090_3epoch_finalonly/final_model/

It disables intermediate checkpoints, so no checkpoint-500/optimizer.pt files are written.
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import gc
import json
import subprocess
from inspect import signature
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean-only train; save final_model only.")
    parser.add_argument("--data_dir", type=str, default="./prepared_data_20k_train_valid_clean_only")
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--valid_file", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="allenai/led-base-16384")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/led_base_clean_runpod4090_3epoch_finalonly")

    parser.add_argument("--max_source_length", type=int, default=2048)
    parser.add_argument("--max_target_length", type=int, default=192)

    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)

    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=500)

    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no_bf16", action="store_false", dest="bf16")
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--no_fp16", action="store_false", dest="fp16")
    parser.add_argument("--tf32", action="store_true", default=True)
    parser.add_argument("--no_tf32", action="store_false", dest="tf32")

    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", action="store_false", dest="gradient_checkpointing")
    parser.add_argument("--auto_find_batch_size", action="store_true", default=False)

    # False avoids safetensors auto-conversion errors for LongT5/LED on some HF-hub versions.
    parser.add_argument("--use_safetensors", action="store_true", default=False)
    parser.add_argument("--no_use_safetensors", action="store_false", dest="use_safetensors")

    parser.add_argument("--minimal_json_loader", action="store_true", default=True)
    parser.add_argument("--hf_json_loader", action="store_false", dest="minimal_json_loader")
    parser.add_argument("--preprocessing_num_workers", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_valid_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--runpod_stop_on_success", action="store_true", default=False)
    parser.add_argument("--runpod_stop_always", action="store_true", default=False)
    return parser.parse_args()


def configure_torch(args):
    if args.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def cuda_supports_bf16() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_bf16_supported())
    except Exception:
        return torch.cuda.get_device_capability()[0] >= 8


def stop_runpod():
    pod_id = os.environ.get("RUNPOD_POD_ID")
    if pod_id:
        subprocess.run(["runpodctl", "pod", "stop", pod_id], check=False)


def resolve_data_files(args) -> Tuple[Path, Path]:
    data_dir = Path(args.data_dir)
    train_file = Path(args.train_file) if args.train_file else data_dir / "train.jsonl"
    valid_file = Path(args.valid_file) if args.valid_file else data_dir / "valid.jsonl"
    if not train_file.exists():
        raise FileNotFoundError(f"Cannot find train file: {train_file}")
    if not valid_file.exists():
        raise FileNotFoundError(f"Cannot find valid file: {valid_file}")
    return train_file, valid_file


def read_jsonl_minimal(path: Path, max_samples=None):
    rows = []
    skipped = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if max_samples is not None and len(rows) >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                x = str(obj.get("input_text", "")).strip()
                y = str(obj.get("target_text", "")).strip()
                if x and y:
                    rows.append({"input_text": x, "target_text": y})
                else:
                    skipped += 1
            except Exception:
                skipped += 1
    if skipped:
        print(f"[WARN] {path}: skipped {skipped:,} bad rows")
    if not rows:
        raise ValueError(f"No valid rows loaded from {path}")
    return rows


def load_data(args, train_file: Path, valid_file: Path) -> DatasetDict:
    if args.minimal_json_loader:
        train_rows = read_jsonl_minimal(train_file, args.max_train_samples)
        valid_rows = read_jsonl_minimal(valid_file, args.max_valid_samples)
        return DatasetDict({
            "train": Dataset.from_list(train_rows),
            "validation": Dataset.from_list(valid_rows),
        })

    raw = load_dataset("json", data_files={"train": str(train_file), "validation": str(valid_file)})
    train_raw = raw["train"]
    valid_raw = raw["validation"]
    if args.max_train_samples is not None:
        train_raw = train_raw.select(range(min(args.max_train_samples, len(train_raw))))
    if args.max_valid_samples is not None:
        valid_raw = valid_raw.select(range(min(args.max_valid_samples, len(valid_raw))))
    return DatasetDict({"train": train_raw, "validation": valid_raw})


def preprocess_dataset(dataset, tokenizer, args, split_name):
    def tokenize_batch(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        inputs = [str(x) if x is not None else "" for x in examples["input_text"]]
        targets = [str(x) if x is not None else "" for x in examples["target_text"]]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, truncation=True, padding=False)
        global_attention_mask = []
        for ids in model_inputs["input_ids"]:
            mask = [0] * len(ids)
            if len(mask) > 0:
                mask[0] = 1
            global_attention_mask.append(mask)
        labels = tokenizer(text_target=targets, max_length=args.max_target_length, truncation=True, padding=False)
        model_inputs["global_attention_mask"] = global_attention_mask
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    num_proc = args.preprocessing_num_workers if args.preprocessing_num_workers > 1 else None
    return dataset.map(
        tokenize_batch,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc=f"Tokenizing {split_name}",
    )


def make_training_args(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    supported = set(signature(Seq2SeqTrainingArguments.__init__).parameters.keys())
    use_bf16 = bool(torch.cuda.is_available() and args.bf16 and cuda_supports_bf16())
    use_fp16 = bool(torch.cuda.is_available() and args.fp16 and not use_bf16)

    candidate = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        predict_with_generate=False,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        auto_find_batch_size=args.auto_find_batch_size,
        optim="adamw_torch",
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        group_by_length=True,
        load_best_model_at_end=False,
        report_to="none",
        save_safetensors=True,
    )

    if "eval_strategy" in supported:
        candidate["eval_strategy"] = "steps"
    elif "evaluation_strategy" in supported:
        candidate["evaluation_strategy"] = "steps"
    else:
        candidate["do_eval"] = True

    # KEY: do not create checkpoint-* folders.
    if "save_strategy" in supported:
        candidate["save_strategy"] = "no"

    if "logging_strategy" in supported:
        candidate["logging_strategy"] = "steps"

    filtered = {k: v for k, v in candidate.items() if k in supported}
    print(f"[INFO] Precision: bf16={use_bf16}, fp16={use_fp16}")
    print(f"[INFO] Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    print("[INFO] Intermediate checkpoints: DISABLED. Only final_model will be saved.")
    return Seq2SeqTrainingArguments(**filtered)


def print_gpu_info():
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"[INFO] CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] VRAM: {p.total_memory / 1024**3:.2f} GB")
        print(f"[INFO] Compute capability: {torch.cuda.get_device_capability(0)}")
        print(f"[INFO] BF16 supported: {cuda_supports_bf16()}")
    else:
        print("[WARN] CUDA is not available.")


def build_trainer(model, training_args, train_dataset, valid_dataset, tokenizer, data_collator):
    supported = set(signature(Seq2SeqTrainer.__init__).parameters.keys())
    candidate = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
    )
    if "tokenizer" in supported:
        candidate["tokenizer"] = tokenizer
    elif "processing_class" in supported:
        candidate["processing_class"] = tokenizer
    return Seq2SeqTrainer(**{k: v for k, v in candidate.items() if k in supported})


def main():
    args = parse_args()
    set_seed(args.seed)
    configure_torch(args)

    try:
        print_gpu_info()
        train_file, valid_file = resolve_data_files(args)
        print(f"[INFO] Train file: {train_file}")
        print(f"[INFO] Valid file: {valid_file}")
        print(f"[INFO] Model: {args.model_name}")
        print(f"[INFO] Epochs: {args.num_train_epochs}")

        raw_data = load_data(args, train_file, valid_file)
        train_raw = raw_data["train"]
        valid_raw = raw_data["validation"]
        print(f"[INFO] Train samples: {len(train_raw):,}")
        print(f"[INFO] Valid samples: {len(valid_raw):,}")

        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, use_safetensors=args.use_safetensors)

        if args.gradient_checkpointing:
            model.config.use_cache = False
            try:
                model.gradient_checkpointing_enable()
            except Exception as e:
                print(f"[WARN] Could not enable gradient checkpointing: {e}")

        train_dataset = preprocess_dataset(train_raw, tokenizer, args, "train")
        valid_dataset = preprocess_dataset(valid_raw, tokenizer, args, "validation")
        del raw_data, train_raw, valid_raw
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding="longest",
            label_pad_token_id=-100,
            pad_to_multiple_of=8 if torch.cuda.is_available() else None,
        )

        training_args = make_training_args(args)
        trainer = build_trainer(model, training_args, train_dataset, valid_dataset, tokenizer, data_collator)

        print("[INFO] Starting clean training: LED-base clean")
        trainer.train()

        final_model_dir = Path(args.output_dir) / "final_model"
        trainer.save_model(str(final_model_dir))
        tokenizer.save_pretrained(str(final_model_dir))

        valid_metrics = trainer.evaluate()
        with (Path(args.output_dir) / "valid_eval_loss_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(valid_metrics, f, ensure_ascii=False, indent=2)

        print("[DONE] Training finished.")
        print(f"[DONE] Final model saved to: {final_model_dir}")

        if args.runpod_stop_on_success or args.runpod_stop_always:
            stop_runpod()

    except Exception as e:
        print(f"[ERROR] Training failed: {type(e).__name__}: {e}")
        if args.runpod_stop_always:
            stop_runpod()
        raise


if __name__ == "__main__":
    main()
