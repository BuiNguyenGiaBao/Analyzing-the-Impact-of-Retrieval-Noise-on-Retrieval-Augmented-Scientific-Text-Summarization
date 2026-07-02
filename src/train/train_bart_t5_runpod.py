#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train BART-base or T5-base for RAG summarization JSONL data on RunPod RTX 4090.

This v2 script is designed for chained experiments:
  1) BART noise-aware
  2) BART clean-matched
  3) T5 noise-aware
  4) T5 clean-matched

Expected JSONL columns:
  - input_text
  - target_text
Optional metadata columns are ignored automatically.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BART/T5 base on JSONL summarization data.")

    # Data
    p.add_argument("--data_dir", type=str, required=True, help="Folder containing JSONL files.")
    p.add_argument("--train_file", type=str, default="train_noiseaware.jsonl")
    p.add_argument("--validation_file", type=str, default="valid_noiseaware.jsonl")
    p.add_argument("--text_column", type=str, default="input_text")
    p.add_argument("--summary_column", type=str, default="target_text")

    # Model
    p.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["facebook/bart-base", "google-t5/t5-base", "t5-base"],
        help="Use facebook/bart-base or google-t5/t5-base.",
    )
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--max_source_length", type=int, default=1024)
    p.add_argument("--max_target_length", type=int, default=512)
    p.add_argument("--generation_max_length", type=int, default=256)
    p.add_argument("--generation_num_beams", type=int, default=4)

    # Training
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--learning_rate", type=float, default=None, help="Default: 3e-5 for BART, 5e-5 for T5.")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--per_device_train_batch_size", type=int, default=None, help="Default: BART=8, T5=4.")
    p.add_argument("--per_device_eval_batch_size", type=int, default=None, help="Default: BART=8, T5=4.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=None, help="Default: BART=2, T5=4.")
    p.add_argument("--gradient_checkpointing", action="store_true", help="Recommended for 1024-token inputs on 24GB GPU.")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--optim", type=str, default="adamw_torch", help="Optimizer name for Transformers TrainingArguments.")

    # Runtime / logging
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--eval_strategy", type=str, default="epoch", choices=["steps", "epoch", "no"])
    p.add_argument("--save_strategy", type=str, default="epoch", choices=["steps", "epoch", "no"])
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--dataloader_num_workers", type=int, default=2)
    p.add_argument(
        "--precision",
        type=str,
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32"],
        help="RTX 4090 supports bf16; auto uses bf16 if available, else fp16 on CUDA.",
    )
    p.add_argument("--report_to", type=str, default="none", help="none, tensorboard, wandb, etc.")
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--dry_run", action="store_true", help="Only load/tokenize a small sample, then exit.")
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=None)
    p.add_argument("--compute_rouge", action="store_true", help="Compute ROUGE during eval. Slower; requires evaluate + rouge_score.")
    p.add_argument("--no_load_best_model_at_end", action="store_true", help="Disable best-checkpoint reload at the end.")

    return p.parse_args()


def normalize_model_name(model_name: str) -> str:
    if model_name == "t5-base":
        return "google-t5/t5-base"
    return model_name


def is_t5(model_name: str) -> bool:
    return "t5" in model_name.lower()


def choose_defaults(args: argparse.Namespace) -> argparse.Namespace:
    model_name = normalize_model_name(args.model_name)
    args.model_name = model_name

    if args.learning_rate is None:
        args.learning_rate = 5e-5 if is_t5(model_name) else 3e-5
    if args.per_device_train_batch_size is None:
        args.per_device_train_batch_size = 4 if is_t5(model_name) else 8
    if args.per_device_eval_batch_size is None:
        args.per_device_eval_batch_size = 4 if is_t5(model_name) else 8
    if args.gradient_accumulation_steps is None:
        args.gradient_accumulation_steps = 4 if is_t5(model_name) else 2
    return args


def get_precision_flags(precision: str) -> Tuple[bool, bool]:
    """Return (fp16, bf16) for TrainingArguments."""
    if precision == "fp32":
        return False, False
    if precision == "fp16":
        return True, False
    if precision == "bf16":
        return False, True

    # auto
    if torch.cuda.is_available():
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return False, True
        return True, False
    return False, False


def safe_count_words(x: Any) -> int:
    return len(str(x or "").replace("\n", " ").split())


def read_jsonl_stats(path: Path, text_col: str, target_col: str, max_scan: int = 200000) -> Dict[str, Any]:
    n = 0
    empty_input = 0
    empty_target = 0
    input_lens: List[int] = []
    target_lens: List[int] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            inp = str(obj.get(text_col, "") or "").strip()
            tar = str(obj.get(target_col, "") or "").strip()
            n += 1
            if not inp:
                empty_input += 1
            if not tar:
                empty_target += 1
            input_lens.append(safe_count_words(inp))
            target_lens.append(safe_count_words(tar))
            if n >= max_scan:
                break

    def stats(vals: List[int]) -> Dict[str, Any]:
        if not vals:
            return {"min": None, "mean": None, "p95": None, "max": None}
        return {
            "min": int(np.min(vals)),
            "mean": float(np.mean(vals)),
            "p95": float(np.percentile(vals, 95)),
            "max": int(np.max(vals)),
        }

    return {
        "path": str(path),
        "scanned_rows": n,
        "empty_input": empty_input,
        "empty_target": empty_target,
        "input_words": stats(input_lens),
        "target_words": stats(target_lens),
    }


def load_jsonl_dataset(args: argparse.Namespace) -> DatasetDict:
    data_dir = Path(args.data_dir)
    train_path = data_dir / args.train_file
    valid_path = data_dir / args.validation_file

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not valid_path.exists():
        raise FileNotFoundError(f"Validation file not found: {valid_path}")

    print("\nDataset quick check:")
    print(json.dumps(read_jsonl_stats(train_path, args.text_column, args.summary_column), indent=2, ensure_ascii=False))
    print(json.dumps(read_jsonl_stats(valid_path, args.text_column, args.summary_column), indent=2, ensure_ascii=False))

    raw = load_dataset(
        "json",
        data_files={"train": str(train_path), "validation": str(valid_path)},
    )

    def non_empty(ex: Dict[str, Any]) -> bool:
        return bool(str(ex.get(args.text_column, "") or "").strip()) and bool(
            str(ex.get(args.summary_column, "") or "").strip()
        )

    raw = raw.filter(non_empty)

    if args.max_train_samples is not None:
        raw["train"] = raw["train"].select(range(min(args.max_train_samples, len(raw["train"]))))
    if args.max_eval_samples is not None:
        raw["validation"] = raw["validation"].select(range(min(args.max_eval_samples, len(raw["validation"]))))

    print(f"\nLoaded rows: train={len(raw['train']):,}, validation={len(raw['validation']):,}")
    if len(raw["train"]) == 0 or len(raw["validation"]) == 0:
        raise ValueError("Dataset is empty after filtering. Check input_text/target_text columns.")
    return raw


def preprocess_dataset(raw: DatasetDict, tokenizer: Any, args: argparse.Namespace) -> DatasetDict:
    text_col = args.text_column
    target_col = args.summary_column

    def preprocess(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        inputs = [str(x or "").strip() for x in batch[text_col]]
        targets = [str(x or "").strip() for x in batch[target_col]]

        model_inputs = tokenizer(
            inputs,
            max_length=args.max_source_length,
            truncation=True,
            padding=False,
        )
        labels = tokenizer(
            text_target=targets,
            max_length=args.max_target_length,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    remove_cols = raw["train"].column_names
    return raw.map(preprocess, batched=True, remove_columns=remove_cols, desc="Tokenizing")


def _strategy_value_for_api(strategy: str) -> str:
    # Transformers uses "no" in newer versions; older APIs still accept it as string.
    return strategy


def build_training_args(args: argparse.Namespace) -> Seq2SeqTrainingArguments:
    fp16, bf16 = get_precision_flags(args.precision)

    sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    eval_key = "eval_strategy" if "eval_strategy" in sig.parameters else "evaluation_strategy"

    # Best-model loading is safe only when both eval and save are enabled and aligned.
    load_best = (
        not args.no_load_best_model_at_end
        and args.eval_strategy != "no"
        and args.save_strategy != "no"
        and args.eval_strategy == args.save_strategy
    )

    kwargs: Dict[str, Any] = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_total_limit": args.save_total_limit,
        "predict_with_generate": True,
        "generation_max_length": args.generation_max_length,
        "generation_num_beams": args.generation_num_beams,
        "fp16": fp16,
        "bf16": bf16,
        "max_grad_norm": args.max_grad_norm,
        "dataloader_num_workers": args.dataloader_num_workers,
        "report_to": [] if args.report_to.lower() == "none" else [args.report_to],
        "load_best_model_at_end": load_best,
        "metric_for_best_model": "eval_loss" if load_best else None,
        "greater_is_better": False if load_best else None,
        "logging_dir": os.path.join(args.output_dir, "logs"),
        "save_strategy": _strategy_value_for_api(args.save_strategy),
        eval_key: _strategy_value_for_api(args.eval_strategy),
        "optim": args.optim,
    }

    # Remove None values for compatibility with older Transformers.
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return Seq2SeqTrainingArguments(**kwargs)


def maybe_build_rouge(tokenizer: Any, args: argparse.Namespace):
    if not args.compute_rouge:
        return None
    try:
        import evaluate  # type: ignore

        rouge = evaluate.load("rouge")
    except Exception as e:
        print(f"[WARN] ROUGE disabled because evaluate/rouge failed to load: {e}")
        return None

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        return {k: round(v * 100, 4) for k, v in result.items()}

    return compute_metrics


def main() -> None:
    args = choose_defaults(parse_args())
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    print("\nRun configuration:")
    print(json.dumps(vars(args), indent=2, ensure_ascii=False))
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"BF16 supported: {getattr(torch.cuda, 'is_bf16_supported', lambda: False)()}")

    raw = load_jsonl_dataset(args)

    print(f"\nLoading tokenizer/model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    tokenized = preprocess_dataset(raw, tokenizer, args)
    if args.dry_run:
        sample = tokenized["train"][0]
        print("\nDry run OK. First tokenized example lengths:")
        print({k: len(v) if isinstance(v, list) else type(v).__name__ for k, v in sample.items()})
        return

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
    )

    training_args = build_training_args(args)
    compute_metrics = maybe_build_rouge(tokenizer, args)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print("\nStarting training...")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print("\nFinal evaluation...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    print(f"\nDone. Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
