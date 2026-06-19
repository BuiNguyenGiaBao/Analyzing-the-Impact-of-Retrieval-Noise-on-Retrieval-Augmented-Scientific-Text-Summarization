#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train PEGASUS-arXiv for RAG scientific summarization.

Model:
    google/pegasus-arxiv

Input JSONL fields:
    input_text
    target_text

This script saves ONLY the final model:
    <output_dir>/final_model

No intermediate checkpoints are saved.
"""

import argparse
import gc
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="google/pegasus-arxiv")
    parser.add_argument("--data_dir", type=str, default="./prepared_data_20k_train_valid_clean_only")
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--valid_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/pegasus_arxiv_clean_runpod4090_3epoch_finalonly")

    parser.add_argument("--max_source_length", type=int, default=1024)
    parser.add_argument("--max_target_length", type=int, default=256)

    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=float, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)

    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=50)

    parser.add_argument("--preprocessing_num_workers", type=int, default=2)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)

    parser.add_argument(
        "--use_safetensors",
        choices=["auto", "true", "false"],
        default="auto",
        help="auto = Transformers tự chọn; true = ép safetensors; false = không ép safetensors.",
    )

    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Bỏ evaluation trong lúc train để tiết kiệm thời gian.",
    )

    return parser.parse_args()


def resolve_files(args):
    if args.train_file is None:
        args.train_file = str(Path(args.data_dir) / "train.jsonl")
    if args.valid_file is None:
        args.valid_file = str(Path(args.data_dir) / "valid.jsonl")

    if not Path(args.train_file).exists():
        raise FileNotFoundError(f"Cannot find train file: {args.train_file}")
    if not args.skip_eval and not Path(args.valid_file).exists():
        raise FileNotFoundError(f"Cannot find valid file: {args.valid_file}")

    return args


def load_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    kwargs = {}
    if args.use_safetensors == "true":
        kwargs["use_safetensors"] = True
    elif args.use_safetensors == "false":
        kwargs["use_safetensors"] = False

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, **kwargs)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    return model, tokenizer


def preprocess_function(examples, tokenizer, args):
    inputs = [str(x) if x is not None else "" for x in examples["input_text"]]
    targets = [str(x) if x is not None else "" for x in examples["target_text"]]

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


def main():
    args = parse_args()
    args = resolve_files(args)
    set_seed(args.seed)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print("=" * 80)
    print("[INFO] Training PEGASUS-arXiv")
    print("[INFO] model_name:", args.model_name)
    print("[INFO] train_file:", args.train_file)
    print("[INFO] valid_file:", args.valid_file if not args.skip_eval else "SKIPPED")
    print("[INFO] output_dir:", args.output_dir)
    print("[INFO] max_source_length:", args.max_source_length)
    print("[INFO] max_target_length:", args.max_target_length)
    print("[INFO] epochs:", args.num_train_epochs)
    print("[INFO] train_bs:", args.per_device_train_batch_size)
    print("[INFO] eval_bs:", args.per_device_eval_batch_size)
    print("[INFO] grad_accum:", args.gradient_accumulation_steps)
    print("[INFO] lr:", args.learning_rate)
    print("[INFO] bf16:", args.bf16)
    print("[INFO] fp16:", args.fp16)
    print("[INFO] gradient_checkpointing:", args.gradient_checkpointing)
    print("=" * 80)

    data_files = {"train": args.train_file}
    if not args.skip_eval:
        data_files["validation"] = args.valid_file

    raw_datasets = load_dataset("json", data_files=data_files)

    model, tokenizer = load_model_and_tokenizer(args)

    remove_columns = raw_datasets["train"].column_names

    tokenized = raw_datasets.map(
        lambda x: preprocess_function(x, tokenizer, args),
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=remove_columns,
        desc="Tokenizing",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    eval_strategy = "no" if args.skip_eval else "epoch"

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,

        do_train=True,
        do_eval=not args.skip_eval,
        evaluation_strategy=eval_strategy,

        save_strategy="no",
        save_total_limit=1,
        load_best_model_at_end=False,

        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,

        logging_steps=args.logging_steps,
        report_to="none",

        predict_with_generate=False,
        generation_max_length=args.max_target_length,

        bf16=args.bf16 and torch.cuda.is_available(),
        fp16=args.fp16 and torch.cuda.is_available(),

        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=True,
        gradient_checkpointing=args.gradient_checkpointing,

        optim="adamw_torch",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation", None),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    final_dir = Path(args.output_dir) / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Saving final model to:", final_dir)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    trainer_state_path = Path(args.output_dir) / "trainer_state.json"
    if trainer.state is not None:
        trainer.state.save_to_json(str(trainer_state_path))

    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("=" * 80)
    print("[DONE] Training complete.")
    print("[DONE] Final model:", final_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()
