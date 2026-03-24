import os
import json
import logging
from typing import Any, Dict, List, Optional
import inspect
import numpy as np
import evaluate
import torch
import argparse
from datasets import Dataset, DatasetDict

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# =========================================================
# Data helpers
# =========================================================

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {path} at line {line_no}: {e}") from e
    return records


def ensure_required_fields(records: List[Dict[str, Any]], path: str) -> None:
    required = {"input_text", "target_text"}
    for idx, rec in enumerate(records):
        missing = required - set(rec.keys())
        if missing:
            raise ValueError(
                f"Missing fields {missing} in record {idx} of {path}. "
                "Each record must contain 'input_text' and 'target_text'."
            )


def maybe_add_split(ds_dict: Dict[str, Dataset], split_name: str, path: Optional[str]) -> None:
    if path and os.path.exists(path):
        records = load_jsonl(path)
        ensure_required_fields(records, path)
        ds_dict[split_name] = Dataset.from_list(records)
        logger.info("Loaded %-16s : %d samples", split_name, len(records))
    else:
        logger.info("Skip %-16s : file not found -> %s", split_name, path)


def build_dataset_dict(
    train_path: str,
    valid_path: str,
    test_clean_path: Optional[str] = None,
    test_noisy_easy_path: Optional[str] = None,
    test_noisy_hard_path: Optional[str] = None,
) -> DatasetDict:
    train_records = load_jsonl(train_path)
    valid_records = load_jsonl(valid_path)

    ensure_required_fields(train_records, train_path)
    ensure_required_fields(valid_records, valid_path)

    ds_dict: Dict[str, Dataset] = {
        "train": Dataset.from_list(train_records),
        "validation": Dataset.from_list(valid_records),
    }

    logger.info("Loaded %-16s : %d samples", "train", len(train_records))
    logger.info("Loaded %-16s : %d samples", "validation", len(valid_records))

    maybe_add_split(ds_dict, "test_clean", test_clean_path)
    maybe_add_split(ds_dict, "test_noisy_easy", test_noisy_easy_path)
    maybe_add_split(ds_dict, "test_noisy_hard", test_noisy_hard_path)

    return DatasetDict(ds_dict)


# =========================================================
# Argument parsing
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VRAM-safe training script for retrieval-augmented summarization on RTX 3050 6GB."
    )

    # Data
    parser.add_argument("--data_dir", type=str, default="./prepared_data")
    parser.add_argument("--train_file", type=str, default="train.jsonl")
    parser.add_argument("--valid_file", type=str, default="valid.jsonl")
    parser.add_argument("--test_clean_file", type=str, default="test_clean.jsonl")
    parser.add_argument("--test_noisy_easy_file", type=str, default="test_noisy_easy.jsonl")
    parser.add_argument("--test_noisy_hard_file", type=str, default="test_noisy_hard.jsonl")

    # Model
    parser.add_argument("--model_name", type=str, default="t5-small")
    parser.add_argument("--output_dir", type=str, default="./outputs_t5_small_3050")
    parser.add_argument("--max_input_length", type=int, default=384)
    parser.add_argument("--max_target_length", type=int, default=96)

    # Training
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=int, default=4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Eval / save
    parser.add_argument("--eval_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--save_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--metric_for_best_model", type=str, default="eval_rougeL")
    # FIX 5: use store_true/store_false pair so --no-greater_is_better can disable it
    parser.add_argument("--greater_is_better", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--early_stopping_patience", type=int, default=2)

    # Generation
    parser.add_argument("--generation_max_length", type=int, default=96)
    parser.add_argument("--generation_num_beams", type=int, default=2)

    # Hardware / misc
    # FIX 1 & 2: use BooleanOptionalAction so flags can be explicitly disabled via --no-fp16, --no-gradient_checkpointing
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval_accumulation_steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_to", type=str, default="none")
    # FIX 6: default to 0 workers to avoid multiprocessing issues on Windows/notebooks
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    return parser.parse_args()


# =========================================================
# Metric helpers
# =========================================================

def save_json(output_dir: str, filename: str, data: Dict[str, Any]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def compute_degradation(
    clean_metrics: Dict[str, Any],
    noisy_metrics: Dict[str, Any],
    prefix: str,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    clean_rouge1 = safe_float(clean_metrics.get("test_clean_rouge1"))
    clean_rouge2 = safe_float(clean_metrics.get("test_clean_rouge2"))
    clean_rougeL = safe_float(clean_metrics.get("test_clean_rougeL"))

    noisy_rouge1 = None
    noisy_rouge2 = None
    noisy_rougeL = None

    for k, v in noisy_metrics.items():
        if k.endswith("_rouge1"):
            noisy_rouge1 = safe_float(v)
        elif k.endswith("_rouge2"):
            noisy_rouge2 = safe_float(v)
        elif k.endswith("_rougeL"):
            noisy_rougeL = safe_float(v)

    def add_drop(name: str, clean_val: Optional[float], noisy_val: Optional[float]) -> None:
        if clean_val is None or noisy_val is None:
            return
        out[f"{prefix}_{name}_abs_drop"] = round(clean_val - noisy_val, 4)
        out[f"{prefix}_{name}_rel_drop_pct"] = round(
            ((clean_val - noisy_val) / max(clean_val, 1e-8)) * 100, 4
        )

    add_drop("rouge1", clean_rouge1, noisy_rouge1)
    add_drop("rouge2", clean_rouge2, noisy_rouge2)
    add_drop("rougeL", clean_rougeL, noisy_rougeL)

    return out


def print_summary(all_metrics: Dict[str, Any]) -> None:
    print("\n" + "=" * 68)
    print("METRICS SUMMARY")
    print("=" * 68)

    for split_name, metrics in all_metrics.items():
        if not isinstance(metrics, dict):
            continue

        rouge1 = None
        rouge2 = None
        rougeL = None

        for k, v in metrics.items():
            if k.endswith("_rouge1"):
                rouge1 = v
            elif k.endswith("_rouge2"):
                rouge2 = v
            elif k.endswith("_rougeL"):
                rougeL = v

        print(f"{split_name:20s} | rouge1={rouge1} | rouge2={rouge2} | rougeL={rougeL}")

    print("=" * 68)


# =========================================================
# Main
# =========================================================

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Torch version      : %s", torch.__version__)
    logger.info("CUDA available     : %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("GPU                : %s", torch.cuda.get_device_name(0))
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info("GPU total memory   : %.2f GB", gpu_mem_gb)

    train_path = os.path.join(args.data_dir, args.train_file)
    valid_path = os.path.join(args.data_dir, args.valid_file)
    test_clean_path = os.path.join(args.data_dir, args.test_clean_file)
    test_noisy_easy_path = os.path.join(args.data_dir, args.test_noisy_easy_file)
    test_noisy_hard_path = os.path.join(args.data_dir, args.test_noisy_hard_file)

    dataset = build_dataset_dict(
        train_path=train_path,
        valid_path=valid_path,
        test_clean_path=test_clean_path,
        test_noisy_easy_path=test_noisy_easy_path,
        test_noisy_hard_path=test_noisy_hard_path,
    )
    logger.info(dataset)

    # FIX 7: consistent use_safetensors=True with summarized.py
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, use_safetensors=True)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # Disable KV-cache when using gradient checkpointing (incompatible)
        model.config.use_cache = False

    # -----------------------------------------------------
    # Tokenization
    # -----------------------------------------------------
    def preprocess_function(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        model_inputs = tokenizer(
            examples["input_text"],
            max_length=args.max_input_length,
            truncation=True,
            padding=False,
        )

        labels = tokenizer(
            text_target=examples["target_text"],
            max_length=args.max_target_length,
            truncation=True,
            padding=False,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = DatasetDict()
    for split_name in dataset.keys():
        tokenized[split_name] = dataset[split_name].map(
            preprocess_function,
            batched=True,
            remove_columns=dataset[split_name].column_names,
            desc=f"Tokenizing {split_name}",
        )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # FIX 3: clamp prediction ids to valid vocab range before decoding
        # to handle any edge-case negative or out-of-range token ids
        predictions = np.clip(predictions, 0, tokenizer.vocab_size - 1)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Replace -100 padding labels with pad_token_id before decoding
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        result = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )
        result = {k: round(v, 4) for k, v in result.items()}

        # Count non-pad tokens per prediction for average generated length
        pred_lens = [
            int(np.count_nonzero(pred != tokenizer.pad_token_id))
            for pred in predictions
        ]
        result["gen_len"] = round(float(np.mean(pred_lens)), 4)

        return result

    # -----------------------------------------------------
    # Data collator
    # -----------------------------------------------------
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        pad_to_multiple_of=8 if (args.fp16 or args.bf16) else None,
    )

    # FIX 4: EarlyStoppingCallback requires load_best_model_at_end=True.
    # Guard: only enable if both eval and save strategies are active.
    load_best = (
        args.eval_strategy != "no"
        and args.save_strategy != "no"
        and args.save_strategy == args.eval_strategy  # strategies must match
    )

    if args.eval_strategy != "no" and args.save_strategy != args.eval_strategy:
        logger.warning(
            "eval_strategy=%s does not match save_strategy=%s. "
            "Disabling load_best_model_at_end and EarlyStoppingCallback "
            "to avoid a TrainingArguments conflict.",
            args.eval_strategy,
            args.save_strategy,
        )

    # -----------------------------------------------------
    # Training arguments
    # -----------------------------------------------------
    sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    accepted = set(sig.parameters.keys())

    kwargs = {
        "output_dir": args.output_dir,
        "do_train": True,
        "do_eval": True,

        # version-dependent names
        "evaluation_strategy": args.eval_strategy,
        "eval_strategy": args.eval_strategy,
        "save_strategy": args.save_strategy,
        "logging_strategy": "steps",
        "logging_steps": args.logging_steps,

        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "eval_accumulation_steps": args.eval_accumulation_steps,

        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "num_train_epochs": args.num_train_epochs,

        "predict_with_generate": True,
        "generation_max_length": args.generation_max_length,
        "generation_num_beams": args.generation_num_beams,

        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": load_best,
        "metric_for_best_model": args.metric_for_best_model,
        "greater_is_better": args.greater_is_better,

        "fp16": args.fp16,
        "bf16": args.bf16,

        "report_to": args.report_to,
        "seed": args.seed,

        "dataloader_pin_memory": True,
        "dataloader_num_workers": args.dataloader_num_workers,
    }

    filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted}

    logger.info("Accepted TrainingArguments keys: %s", sorted(filtered_kwargs.keys()))

    training_args = Seq2SeqTrainingArguments(**filtered_kwargs)

    callbacks = []
    # FIX 4: only attach EarlyStoppingCallback when load_best_model_at_end=True
    if load_best:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            )
        )

    trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=callbacks,
)

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------
    logger.info("Starting training...")
    train_result = trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    all_metrics: Dict[str, Any] = {}
    all_metrics["train"] = train_result.metrics
    save_json(args.output_dir, "train_metrics.json", train_result.metrics)

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------
    logger.info("Running validation evaluation...")
    valid_metrics = trainer.evaluate(
        eval_dataset=tokenized["validation"],
        metric_key_prefix="eval",
    )
    all_metrics["validation"] = valid_metrics
    save_json(args.output_dir, "validation_metrics.json", valid_metrics)

    # -----------------------------------------------------
    # Test clean
    # -----------------------------------------------------
    if "test_clean" in tokenized:
        logger.info("Running test_clean evaluation...")
        test_clean_metrics = trainer.evaluate(
            eval_dataset=tokenized["test_clean"],
            metric_key_prefix="test_clean",
        )
        all_metrics["test_clean"] = test_clean_metrics
        save_json(args.output_dir, "test_clean_metrics.json", test_clean_metrics)
    else:
        test_clean_metrics = {}

    # -----------------------------------------------------
    # Test noisy easy
    # -----------------------------------------------------
    if "test_noisy_easy" in tokenized:
        logger.info("Running test_noisy_easy evaluation...")
        test_noisy_easy_metrics = trainer.evaluate(
            eval_dataset=tokenized["test_noisy_easy"],
            metric_key_prefix="test_noisy_easy",
        )
        all_metrics["test_noisy_easy"] = test_noisy_easy_metrics
        save_json(args.output_dir, "test_noisy_easy_metrics.json", test_noisy_easy_metrics)

        if test_clean_metrics:
            easy_drop = compute_degradation(
                clean_metrics=test_clean_metrics,
                noisy_metrics=test_noisy_easy_metrics,
                prefix="clean_to_easy",
            )
            all_metrics["degradation_easy"] = easy_drop
            save_json(args.output_dir, "degradation_easy.json", easy_drop)

    # -----------------------------------------------------
    # Test noisy hard
    # -----------------------------------------------------
    if "test_noisy_hard" in tokenized:
        logger.info("Running test_noisy_hard evaluation...")
        test_noisy_hard_metrics = trainer.evaluate(
            eval_dataset=tokenized["test_noisy_hard"],
            metric_key_prefix="test_noisy_hard",
        )
        all_metrics["test_noisy_hard"] = test_noisy_hard_metrics
        save_json(args.output_dir, "test_noisy_hard_metrics.json", test_noisy_hard_metrics)

        if test_clean_metrics:
            hard_drop = compute_degradation(
                clean_metrics=test_clean_metrics,
                noisy_metrics=test_noisy_hard_metrics,
                prefix="clean_to_hard",
            )
            all_metrics["degradation_hard"] = hard_drop
            save_json(args.output_dir, "degradation_hard.json", hard_drop)

    save_json(args.output_dir, "all_metrics.json", all_metrics)
    print_summary(all_metrics)

    logger.info("Training finished.")
    logger.info("Model saved to: %s", args.output_dir)


if __name__ == "__main__":
    main()