#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FINAL RTX3050 local evaluator for RAG retrieval-noise summarization.

Chạy được với layout Windows kiểu:

D:\đồ án\downloads\
  eval_rag_noise_metrics_rtx3050_local_FINAL.py
  prepared_data\
    test_clean.jsonl
    test_noisy_easy.jsonl
    test_noisy_hard.jsonl

  t5_base_clean_runpod4090_3epoch_finalonly\final_model\
  bart_base_clean_runpod4090_3epoch_finalonly\final_model\
  longt5_base_clean_runpod4090_3epoch_finalonly\final_model\
  led_base_clean_runpod4090_3epoch_finalonly\final_model\

  t5_base_final_model.tar.gz
  bart_base_final_model.tar.gz
  longt5_base_final_model.tar.gz

Metric:
- ROUGE-1 / ROUGE-2 / ROUGE-L
- BERTScore Precision / Recall / F1 nếu không dùng --skip_bertscore
- retrieval_precision@K / retrieval_recall@K nếu có trong JSONL
- context_precision@K / context_recall@K nếu có trong JSONL
- noise_ratio
- degradation
- robustness_gain
- clean_tradeoff
- paired bootstrap 95% CI
- paired t-test p-value nếu có scipy
"""

import argparse
import gc
import json
import os
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent


MODEL_SPECS = {
    "t5_base": {
        "display": "T5-base",
        "base_tokenizer": "t5-base",
        "clean_folders": [
            "t5_base_clean_runpod4090_3epoch_finalonly",
            "t5_base_clean_rtx3050_3epoch_finalonly",
        ],
        "noise_folders": [
            "t5_base_runpod4090_1epoch",
            "t5_base_rtx3050_1epoch",
        ],
        "noise_tar": "t5_base_final_model.tar.gz",
        "max_source_length": 1024,
        "max_new_tokens": 256,
    },
    "bart_base": {
        "display": "BART-base",
        "base_tokenizer": "facebook/bart-base",
        "clean_folders": [
            "bart_base_clean_runpod4090_3epoch_finalonly",
            "bart_base_clean_rtx3050_3epoch_finalonly",
        ],
        "noise_folders": [
            "bart_base_runpod4090_1epoch",
            "bart_base_rtx3050_1epoch",
        ],
        "noise_tar": "bart_base_final_model.tar.gz",
        "max_source_length": 1024,
        "max_new_tokens": 256,
    },
    "longt5_base": {
        "display": "LongT5-base",
        "base_tokenizer": "google/long-t5-tglobal-base",
        "clean_folders": [
            "longt5_base_clean_runpod4090_3epoch_finalonly",
            "longt5_base_clean_rtx3050_3epoch_finalonly",
        ],
        "noise_folders": [
            "longt5_base_runpod4090_1epoch",
            "longt5_base_rtx3050_1epoch",
        ],
        "noise_tar": "longt5_base_final_model.tar.gz",
        "max_source_length": 1024,
        "max_new_tokens": 256,
    },
    "led_base": {
        "display": "LED-base",
        "base_tokenizer": "allenai/led-base-16384",
        "clean_folders": [
            "led_base_clean_runpod4090_3epoch_finalonly",
            "led_base_clean_rtx3050_3epoch_finalonly",
        ],
        "noise_folders": [
            "led_base_runpod4090_1epoch",
            "led_base_rtx3050_1epoch",
        ],
        "noise_tar": "led_base_final_model.tar.gz",
        "max_source_length": 2048,
        "max_new_tokens": 192,
    },
}


WEIGHT_FILES = [
    "pytorch_model.bin",
    "model.safetensors",
    "tf_model.h5",
    "model.ckpt.index",
    "flax_model.msgpack",
    "pytorch_model.bin.index.json",
    "model.safetensors.index.json",
]


def as_abs(path_like) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else SCRIPT_DIR / p


def has_model_weights(folder: Path) -> bool:
    return any((folder / name).exists() for name in WEIGHT_FILES)


def is_hf_model_folder(folder: Path) -> bool:
    return folder.exists() and folder.is_dir() and (folder / "config.json").exists() and has_model_weights(folder)


def resolve_hf_model_folder(folder: Path) -> Optional[Path]:
    """
    Chấp nhận:
    - folder/config.json + model weights
    - folder/final_model/config.json + model weights
    - nested subfolder có config.json + model weights
    """
    folder = as_abs(folder)

    if is_hf_model_folder(folder):
        return folder

    final_model = folder / "final_model"
    if is_hf_model_folder(final_model):
        return final_model

    if folder.exists() and folder.is_dir():
        for sub in folder.rglob("*"):
            if sub.is_dir() and is_hf_model_folder(sub):
                return sub

    return None


def invalid_reason(folder: Path) -> str:
    folder = as_abs(folder)
    if not folder.exists():
        return f"NOT EXISTS: {folder}"
    if not folder.is_dir():
        return f"NOT FOLDER: {folder}"
    if not (folder / "config.json").exists():
        return f"MISSING config.json: {folder}"
    if not has_model_weights(folder):
        return f"MISSING model weights: {folder}"
    return f"OK: {folder}"


def safe_extract_tar_gz(tar_path: Path, dest: Path) -> None:
    tar_path = as_abs(tar_path)
    dest = as_abs(dest)
    dest.mkdir(parents=True, exist_ok=True)

    def safe_member(member: tarfile.TarInfo) -> bool:
        target = (dest / member.name).resolve()
        return str(target).startswith(str(dest.resolve()))

    print(f"[INFO] Extracting {tar_path} -> {dest}")
    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()
        safe_members = [m for m in members if safe_member(m)]
        if len(safe_members) != len(members):
            raise RuntimeError(f"Unsafe tar.gz member found in {tar_path}")
        tar.extractall(dest, members=safe_members)


def resolve_clean_model(model_key: str) -> Optional[Path]:
    spec = MODEL_SPECS[model_key]
    candidates = []

    for folder in spec["clean_folders"]:
        candidates.append(as_abs(folder))
        candidates.append(as_abs("checkpoints") / folder)

    for c in candidates:
        found = resolve_hf_model_folder(c)
        if found is not None:
            return found

    return None


def resolve_noise_model(model_key: str) -> Optional[Path]:
    spec = MODEL_SPECS[model_key]
    candidates = []

    for folder in spec["noise_folders"]:
        candidates.append(as_abs(folder))
        candidates.append(as_abs("checkpoints") / folder)

    for c in candidates:
        found = resolve_hf_model_folder(c)
        if found is not None:
            return found

    tar_path = as_abs(spec["noise_tar"])
    if tar_path.exists():
        dest = as_abs("checkpoints") / spec["noise_folders"][0]
        safe_extract_tar_gz(tar_path, dest)

        found = resolve_hf_model_folder(dest)
        if found is not None:
            return found

    return None


def resolve_data_dir(data_dir_arg: str) -> Path:
    candidates = []
    p = Path(data_dir_arg)

    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(SCRIPT_DIR / p)
        candidates.append(Path.cwd() / p)

    for c in candidates:
        if (
            (c / "test_clean.jsonl").exists()
            and (c / "test_noisy_easy.jsonl").exists()
            and (c / "test_noisy_hard.jsonl").exists()
        ):
            return c

    return candidates[0]


def guess_base_tokenizer_name(model_key: str, model_dir: Path) -> str:
    if model_key in MODEL_SPECS:
        return MODEL_SPECS[model_key]["base_tokenizer"]

    s = str(model_dir).replace("\\", "/").lower()
    if "longt5" in s or "long-t5" in s:
        return "google/long-t5-tglobal-base"
    if "bart" in s:
        return "facebook/bart-base"
    if "led" in s:
        return "allenai/led-base-16384"
    if "t5" in s:
        return "t5-base"
    return str(model_dir)


def load_tokenizer_with_fallback(model_key: str, model_dir: Path):
    """
    Thử load tokenizer từ final_model.
    Nếu final_model thiếu tokenizer, dùng tokenizer gốc theo model.
    """
    try:
        return AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    except Exception:
        base_name = guess_base_tokenizer_name(model_key, model_dir)
        print(f"[WARN] Không load được tokenizer từ local folder: {model_dir}")
        print(f"[WARN] Dùng tokenizer gốc thay thế: {base_name}")
        print("[WARN] Nếu lần đầu chạy local, máy cần internet để tải tokenizer gốc.")
        return AutoTokenizer.from_pretrained(base_name, use_fast=True)


def load_model(model_dir: Path, use_safetensors_mode: str):
    """
    use_safetensors_mode:
    - auto: để transformers tự chọn .bin hoặc .safetensors
    - true: ép dùng safetensors
    - false: không ép safetensors
    """
    kwargs = {}
    if use_safetensors_mode == "true":
        kwargs["use_safetensors"] = True
    elif use_safetensors_mode == "false":
        kwargs["use_safetensors"] = False

    try:
        return AutoModelForSeq2SeqLM.from_pretrained(str(model_dir), **kwargs)
    except OSError as e:
        raise OSError(
            f"Không load được model từ folder: {model_dir}\n"
            f"Folder phải có config.json và một file weight như: {', '.join(WEIGHT_FILES)}\n"
            f"Kiểm tra bằng lệnh PowerShell:\n"
            f"dir /s /b \"{model_dir}\\*model*\"\n"
            f"dir /s /b \"{model_dir}\\*.safetensors\"\n"
        ) from e


def print_debug_paths(args) -> None:
    print("=" * 90)
    print("[DEBUG PATHS]")
    print("SCRIPT_DIR:", SCRIPT_DIR)
    print("CWD       :", Path.cwd())
    print("DATA_DIR  :", resolve_data_dir(args.data_dir))
    print()

    for key, spec in MODEL_SPECS.items():
        clean = resolve_clean_model(key)
        noise = resolve_noise_model(key)

        print(f"{key} / {spec['display']}")
        print("  clean resolved:", clean if clean else "NOT FOUND")
        print("  noise resolved:", noise if noise else "NOT FOUND")

        print("  clean candidates:")
        for folder in spec["clean_folders"]:
            for c in [as_abs(folder), as_abs("checkpoints") / folder]:
                print("   -", invalid_reason(c))
                fm = c / "final_model"
                if fm.exists():
                    print("   -", invalid_reason(fm))

        print("  noise candidates:")
        for folder in spec["noise_folders"]:
            for c in [as_abs(folder), as_abs("checkpoints") / folder]:
                print("   -", invalid_reason(c))
                fm = c / "final_model"
                if fm.exists():
                    print("   -", invalid_reason(fm))

        tar_path = as_abs(spec["noise_tar"])
        print("  noise tar:", tar_path if tar_path.exists() else f"NOT FOUND: {tar_path}")
        print()

    print("=" * 90)


def read_jsonl(path: Path, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find test file: {path}")

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if max_samples is not None and len(rows) >= max_samples:
                break

            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            if obj.get("input_text") and obj.get("target_text"):
                rows.append(obj)

    if not rows:
        raise ValueError(f"No valid samples in {path}")

    return rows


def safe_mean(values) -> float:
    vals = []
    for v in values:
        try:
            if v is not None and not pd.isna(v):
                vals.append(float(v))
        except Exception:
            pass
    return float(np.mean(vals)) if vals else float("nan")


def extract_meta(record: Dict[str, Any]) -> Dict[str, Any]:
    keep = [
        "paper_id", "doc_id", "source_doc_id", "split", "sample_type",
        "noise_mode", "noise_source", "num_noise_chunks", "num_contexts",
        "num_context_chunks_eval", "num_documents_in_context",
        "num_retrieved_chunks", "num_relevant_chunks", "num_chunks_total_in_paper",
    ]

    item = {k: record.get(k) for k in keep if k in record}

    for k, v in record.items():
        if (
            k.startswith("retrieval_precision@")
            or k.startswith("retrieval_recall@")
            or k.startswith("context_precision@")
            or k.startswith("context_recall@")
            or k.startswith("precision@")
            or k.startswith("recall@")
            or k.startswith("hits@")
            or k.startswith("retrieval_hits@")
            or k.startswith("context_hits@")
        ):
            item[k] = v

    if "paper_id" not in item:
        item["paper_id"] = record.get("source_doc_id", record.get("doc_id", None))

    num_noise = record.get("num_noise_chunks", 0) or 0
    num_contexts = record.get("num_contexts", None)

    if num_contexts is None and isinstance(record.get("context_chunk_ids"), list):
        num_contexts = len(record["context_chunk_ids"])

    item["noise_ratio"] = float(num_noise) / float(num_contexts) if num_contexts and num_contexts > 0 else 0.0
    return item


@torch.no_grad()
def generate_predictions(
    model_key: str,
    model_dir: Path,
    records: List[Dict[str, Any]],
    args,
    max_source_length: int,
    max_new_tokens: int,
) -> List[str]:
    print(f"[INFO] Loading tokenizer/model from: {model_dir}")

    tokenizer = load_tokenizer_with_fallback(model_key, model_dir)
    model = load_model(model_dir, args.use_safetensors)

    model.to(args.device)
    model.eval()

    preds = []
    use_amp = args.device.startswith("cuda") and args.fp16

    for start in tqdm(range(0, len(records), args.gen_batch_size), desc="Generating", unit="batch"):
        batch = records[start:start + args.gen_batch_size]
        texts = [str(x.get("input_text", "")) for x in batch]

        enc = tokenizer(
            texts,
            max_length=max_source_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(args.device) for k, v in enc.items()}

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                ids = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    num_beams=args.num_beams,
                    length_penalty=1.0,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )
        else:
            ids = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                num_beams=args.num_beams,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        preds.extend(tokenizer.batch_decode(ids, skip_special_tokens=True))

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return [p.strip() for p in preds]


def compute_rouge(preds: List[str], refs: List[str]) -> pd.DataFrame:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )

    rows = []
    for pred, ref in tqdm(list(zip(preds, refs)), desc="ROUGE", unit="sample"):
        s = scorer.score(ref, pred)
        rows.append({
            "rouge1": s["rouge1"].fmeasure,
            "rouge2": s["rouge2"].fmeasure,
            "rougeL": s["rougeL"].fmeasure,
        })

    return pd.DataFrame(rows)


def compute_bertscore(preds: List[str], refs: List[str], args) -> pd.DataFrame:
    from bert_score import score as bertscore_score

    bert_device = args.bertscore_device
    if bert_device == "auto":
        bert_device = args.device

    print(f"[INFO] BERTScore model={args.bertscore_model_type}, device={bert_device}, batch={args.bertscore_batch_size}")

    P, R, F1 = bertscore_score(
        cands=preds,
        refs=refs,
        model_type=args.bertscore_model_type,
        batch_size=args.bertscore_batch_size,
        device=bert_device,
        lang=args.bertscore_lang,
        verbose=True,
        rescale_with_baseline=False,
    )

    return pd.DataFrame({
        "bertscore_precision": P.cpu().numpy().tolist(),
        "bertscore_recall": R.cpu().numpy().tolist(),
        "bertscore_f1": F1.cpu().numpy().tolist(),
    })


def evaluate_one(
    model_key: str,
    display_name: str,
    train_setting: str,
    model_dir: Path,
    condition: str,
    records: List[Dict[str, Any]],
    args,
    max_source_length: int,
    max_new_tokens: int,
) -> pd.DataFrame:
    print("-" * 90)
    print(f"[INFO] Evaluating {display_name} | {train_setting} | {condition}")

    preds = generate_predictions(
        model_key=model_key,
        model_dir=model_dir,
        records=records,
        args=args,
        max_source_length=max_source_length,
        max_new_tokens=max_new_tokens,
    )

    refs = [str(x.get("target_text", "")) for x in records]

    df = pd.DataFrame([extract_meta(x) for x in records])
    df["model_label"] = model_key
    df["train_setting"] = train_setting
    df["condition"] = condition
    df["reference"] = refs
    df["prediction"] = preds

    df = pd.concat([df.reset_index(drop=True), compute_rouge(preds, refs).reset_index(drop=True)], axis=1)

    if not args.skip_bertscore:
        df = pd.concat([df.reset_index(drop=True), compute_bertscore(preds, refs, args).reset_index(drop=True)], axis=1)

    return df


def metric_columns(df: pd.DataFrame) -> List[str]:
    base = [
        "rouge1", "rouge2", "rougeL",
        "bertscore_precision", "bertscore_recall", "bertscore_f1",
    ]

    retrieval = [
        c for c in df.columns
        if c.startswith((
            "retrieval_precision@",
            "retrieval_recall@",
            "context_precision@",
            "context_recall@",
            "precision@",
            "recall@",
        ))
        or c == "noise_ratio"
    ]

    return [c for c in base if c in df.columns] + sorted(retrieval)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(["model_label", "train_setting", "condition"]):
        row = {
            "model_label": keys[0],
            "train_setting": keys[1],
            "condition": keys[2],
            "n_samples": int(len(g)),
        }
        for c in metric_columns(df):
            row[c] = safe_mean(g[c])
        rows.append(row)

    return pd.DataFrame(rows)


def bootstrap_ci(diffs, n_boot=1000, seed=42):
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]

    if len(diffs) == 0:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    means = [
        float(np.mean(rng.choice(diffs, size=len(diffs), replace=True)))
        for _ in range(n_boot)
    ]

    return (
        float(np.mean(diffs)),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def paired_ttest_p(diffs):
    try:
        from scipy import stats

        diffs = np.asarray(diffs, dtype=float)
        diffs = diffs[~np.isnan(diffs)]

        if len(diffs) < 2:
            return float("nan")

        return float(stats.ttest_1samp(diffs, popmean=0.0).pvalue)
    except Exception:
        return float("nan")


def paired_stats(df_a: pd.DataFrame, df_b: pd.DataFrame, metric: str, args) -> Dict[str, Any]:
    if args.id_col not in df_a.columns or args.id_col not in df_b.columns:
        return {
            "metric": metric,
            "n_pairs": 0,
            "mean_diff": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "paired_ttest_p": float("nan"),
            "note": f"missing id_col={args.id_col}",
        }

    a = df_a[[args.id_col, metric]].dropna()
    b = df_b[[args.id_col, metric]].dropna()
    merged = a.merge(b, on=args.id_col, suffixes=("_a", "_b"))

    if merged.empty:
        return {
            "metric": metric,
            "n_pairs": 0,
            "mean_diff": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "paired_ttest_p": float("nan"),
            "note": "no matched pairs",
        }

    diffs = merged[f"{metric}_a"].to_numpy() - merged[f"{metric}_b"].to_numpy()
    mean, lo, hi = bootstrap_ci(diffs, n_boot=args.n_bootstrap, seed=args.seed)

    return {
        "metric": metric,
        "n_pairs": int(len(merged)),
        "mean_diff": mean,
        "ci95_low": lo,
        "ci95_high": hi,
        "paired_ttest_p": paired_ttest_p(diffs),
    }


def text_metrics(df: pd.DataFrame) -> List[str]:
    return [m for m in ["rouge1", "rouge2", "rougeL", "bertscore_f1"] if m in df.columns]


def degradation_table(df: pd.DataFrame, args) -> pd.DataFrame:
    rows = []
    for (model_label, train_setting), g in df.groupby(["model_label", "train_setting"]):
        clean = g[g["condition"] == "clean"]

        if clean.empty:
            continue

        for cond in ["easy", "hard"]:
            noisy = g[g["condition"] == cond]

            if noisy.empty:
                continue

            for metric in text_metrics(df):
                stat = paired_stats(clean, noisy, metric, args)
                stat.update({
                    "analysis": "degradation",
                    "model_label": model_label,
                    "train_setting": train_setting,
                    "comparison": f"clean_minus_{cond}",
                    "interpretation": "positive means performance drops under noise",
                })
                rows.append(stat)

    return pd.DataFrame(rows)


def robustness_gain_table(df: pd.DataFrame, args) -> pd.DataFrame:
    rows = []
    for (model_label, condition), g in df.groupby(["model_label", "condition"]):
        clean_model = g[g["train_setting"] == "clean_trained"]
        noise_model = g[g["train_setting"] == "noise_aware"]

        if clean_model.empty or noise_model.empty:
            continue

        for metric in text_metrics(df):
            stat = paired_stats(noise_model, clean_model, metric, args)
            stat.update({
                "analysis": "robustness_gain",
                "model_label": model_label,
                "condition": condition,
                "comparison": "noise_aware_minus_clean_trained",
                "interpretation": "positive means noise-aware performs better",
            })
            rows.append(stat)

    return pd.DataFrame(rows)


def clean_tradeoff_table(df: pd.DataFrame, args) -> pd.DataFrame:
    rows = []
    for model_label, g_all in df.groupby("model_label"):
        g = g_all[g_all["condition"] == "clean"]
        clean_model = g[g["train_setting"] == "clean_trained"]
        noise_model = g[g["train_setting"] == "noise_aware"]

        if clean_model.empty or noise_model.empty:
            continue

        for metric in text_metrics(df):
            stat = paired_stats(noise_model, clean_model, metric, args)
            stat.update({
                "analysis": "clean_tradeoff",
                "model_label": model_label,
                "condition": "clean",
                "comparison": "noise_aware_minus_clean_trained_on_clean_test",
                "interpretation": "negative means noise training hurts clean performance",
            })
            rows.append(stat)

    return pd.DataFrame(rows)


def evaluate_model_key(model_key: str, args) -> Optional[pd.DataFrame]:
    spec = MODEL_SPECS[model_key]

    if args.clean_model_dir:
        clean_dir = resolve_hf_model_folder(as_abs(args.clean_model_dir))
    else:
        clean_dir = resolve_clean_model(model_key)

    if args.noise_model_dir:
        noise_dir = resolve_hf_model_folder(as_abs(args.noise_model_dir))
    else:
        noise_dir = resolve_noise_model(model_key)

    max_source_length = args.max_source_length or spec["max_source_length"]
    max_new_tokens = args.max_new_tokens or spec["max_new_tokens"]

    print("=" * 90)
    print(f"[INFO] Model: {spec['display']}")
    print(f"[INFO] clean_model_dir: {clean_dir if clean_dir else 'NOT FOUND'}")
    print(f"[INFO] noise_model_dir: {noise_dir if noise_dir else 'NOT FOUND'}")
    print(f"[INFO] max_source_length={max_source_length}, max_new_tokens={max_new_tokens}")
    print("=" * 90)

    model_pairs = []

    if not args.only_noise_model:
        if clean_dir is not None:
            model_pairs.append(("clean_trained", clean_dir))
        else:
            print(f"[WARN] Clean model not found for {model_key}")

    if not args.only_clean_model:
        if noise_dir is not None:
            model_pairs.append(("noise_aware", noise_dir))
        else:
            print(f"[WARN] Noise-aware model not found for {model_key}")

    if not model_pairs:
        print(f"[WARN] No model found for {spec['display']}. Skipped.")
        return None

    data_dir = resolve_data_dir(args.data_dir)
    tests = [
        ("clean", data_dir / "test_clean.jsonl"),
        ("easy", data_dir / "test_noisy_easy.jsonl"),
        ("hard", data_dir / "test_noisy_hard.jsonl"),
    ]

    missing_tests = [str(p) for _, p in tests if not p.exists()]
    if missing_tests:
        print("[WARN] Missing test files:")
        for p in missing_tests:
            print("  ", p)
        return None

    model_out = as_abs(args.output_root) / model_key
    model_out.mkdir(parents=True, exist_ok=True)

    all_dfs = []

    for condition, test_file in tests:
        records = read_jsonl(test_file, args.max_samples)
        print(f"[INFO] Loaded {len(records):,} samples for condition={condition}")

        for train_setting, model_dir in model_pairs:
            df = evaluate_one(
                model_key=model_key,
                display_name=spec["display"],
                train_setting=train_setting,
                model_dir=model_dir,
                condition=condition,
                records=records,
                args=args,
                max_source_length=max_source_length,
                max_new_tokens=max_new_tokens,
            )

            per_csv = model_out / f"per_sample_{model_key}_{train_setting}_{condition}.csv"
            df.to_csv(per_csv, index=False)
            print("[DONE] Saved:", per_csv)

            all_dfs.append(df)

    if not all_dfs:
        return None

    full = pd.concat(all_dfs, ignore_index=True)

    full.to_csv(model_out / f"{model_key}_all_per_sample_metrics.csv", index=False)

    with (model_out / f"{model_key}_all_per_sample_metrics.jsonl").open("w", encoding="utf-8") as f:
        for row in full.to_dict(orient="records"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_table(full).to_csv(model_out / f"{model_key}_summary_metrics.csv", index=False)
    degradation_table(full, args).to_csv(model_out / f"{model_key}_degradation_stats.csv", index=False)
    robustness_gain_table(full, args).to_csv(model_out / f"{model_key}_robustness_gain_stats.csv", index=False)
    clean_tradeoff_table(full, args).to_csv(model_out / f"{model_key}_clean_tradeoff_stats.csv", index=False)

    print("[DONE] Finished model:", spec["display"])
    print("[DONE] Output folder:", model_out)

    return full


def parse_args():
    p = argparse.ArgumentParser(description="FINAL RTX3050 local evaluator for RAG noise experiment.")

    p.add_argument("--model", default="all", choices=["all", "t5_base", "bart_base", "longt5_base", "led_base"])
    p.add_argument("--data_dir", default="./prepared_data")
    p.add_argument("--output_root", default="./metrics_rtx3050")

    p.add_argument("--clean_model_dir", default=None, help="Override clean model dir. Use only when --model is not all.")
    p.add_argument("--noise_model_dir", default=None, help="Override noise model dir. Use only when --model is not all.")

    p.add_argument("--max_source_length", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=None)

    p.add_argument("--gen_batch_size", type=int, default=1)
    p.add_argument("--num_beams", type=int, default=2)

    p.add_argument("--skip_bertscore", action="store_true")
    p.add_argument("--bertscore_model_type", default="roberta-base")
    p.add_argument("--bertscore_batch_size", type=int, default=2)
    p.add_argument("--bertscore_device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--bertscore_lang", default="en")

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--no_fp16", action="store_false", dest="fp16")

    p.add_argument(
        "--use_safetensors",
        default="auto",
        choices=["auto", "true", "false"],
        help="auto = transformers tự chọn; true = ép safetensors; false = không ép safetensors",
    )

    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--n_bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--id_col", default="paper_id")

    p.add_argument("--only_clean_model", action="store_true")
    p.add_argument("--only_noise_model", action="store_true")
    p.add_argument("--debug_paths", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print("[INFO] Device:", args.device)
    print("[INFO] FP16:", args.fp16)
    print("[INFO] Generation batch size:", args.gen_batch_size)
    print("[INFO] Num beams:", args.num_beams)
    print("[INFO] use_safetensors:", args.use_safetensors)

    if args.debug_paths:
        print_debug_paths(args)
        return

    keys = list(MODEL_SPECS.keys()) if args.model == "all" else [args.model]

    all_results = []

    for key in keys:
        result = evaluate_model_key(key, args)
        if result is not None:
            all_results.append(result)

    if not all_results:
        print_debug_paths(args)
        raise RuntimeError("No results produced. Check model paths and test files above.")

    combined = pd.concat(all_results, ignore_index=True)

    out_root = as_abs(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    combined.to_csv(out_root / "all_models_all_per_sample_metrics.csv", index=False)
    summary_table(combined).to_csv(out_root / "all_models_summary_metrics.csv", index=False)
    degradation_table(combined, args).to_csv(out_root / "all_models_degradation_stats.csv", index=False)
    robustness_gain_table(combined, args).to_csv(out_root / "all_models_robustness_gain_stats.csv", index=False)
    clean_tradeoff_table(combined, args).to_csv(out_root / "all_models_clean_tradeoff_stats.csv", index=False)

    print("=" * 90)
    print("[DONE] ALL FINISHED")
    print("[DONE] Combined output:", out_root)
    print("=" * 90)


if __name__ == "__main__":
    main()
