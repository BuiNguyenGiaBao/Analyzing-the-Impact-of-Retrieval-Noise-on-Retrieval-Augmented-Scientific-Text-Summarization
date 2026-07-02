# Truncation-Aware Evaluation Protocol for Noise-Aware RAG Summarization

## Overview

This repository contains the experimental pipeline, dataset construction scripts, evaluation metrics, and manuscript files for a study on **truncation-aware robustness evaluation** in retrieval-augmented generation (RAG) summarization.

The core contribution is not a new summarization model. Instead, this project proposes an **evaluation protocol** for deciding when a robustness claim under retrieval noise is reliable. In particular, the protocol checks whether apparent robustness is genuine or whether it may be inflated by input truncation, especially under additive-noise settings.

## Vietnamese Summary

Kho lưu trữ này chứa mã nguồn, quy trình tạo dữ liệu, script đánh giá và bản thảo bài báo cho nghiên cứu về **giao thức đánh giá robustness có xét đến truncation** trong bài toán tóm tắt văn bản khoa học bằng RAG.

Đóng góp chính của nghiên cứu không phải là một mô hình mới, mà là một **khung đánh giá** giúp xác định khi nào một tuyên bố “mô hình chống nhiễu tốt” là đáng tin cậy. Điểm quan trọng là tách bạch giữa:

- mô hình thật sự robust với nhiễu truy hồi;
- mô hình có vẻ robust vì phần nhiễu đã bị cắt mất do giới hạn độ dài đầu vào.

## Research Motivation

RAG summarization systems can be affected by noisy retrieved context. However, robustness evaluation is often confounded by input length. When additive noise is appended to clean context, the input becomes longer and may exceed the model maximum source length. If noisy chunks are truncated before reaching the encoder, high robustness scores may be misleading.

This project addresses that issue by introducing a truncation-aware evaluation protocol that combines:

1. clean-context evaluation;
2. additive-noise evaluation;
3. substitutive-noise evaluation;
4. retrieval diagnostics;
5. model-specific truncation diagnostics;
6. robustness degradation analysis;
7. statistical testing;
8. net utility analysis.

## Main Contribution

The main contribution is a **truncation-aware evaluation framework** for noise-aware RAG summarization.

The framework evaluates whether robustness claims remain valid after checking:

- whether retrieved distractors are actually present in the final context;
- whether the input is heavily truncated;
- whether additive-noise gains persist under substitutive noise;
- whether performance gains are practically meaningful after accounting for clean-condition degradation;
- whether results are supported by paired statistical tests.

## Experimental Design

### Models

The main experiments use:

- `facebook/bart-base`
- `google-t5/t5-base`

Each architecture has two variants:

- **clean-matched baseline**: trained on clean contexts;
- **noise-aware model**: trained on clean and noisy contexts.

BART-base is used as the main case study because its truncation profile is more stable. T5-base is treated as a diagnostic baseline, especially because T5-base additive-noise inputs can suffer from high truncation.

### Data Conditions

The evaluation uses five test conditions:

| Condition | Description |
|---|---|
| `test_clean` | Clean retrieved context only |
| `test_noisy_easy_additive` | Clean context plus easy distractor chunks |
| `test_noisy_hard_additive` | Clean context plus hard distractor chunks |
| `test_noisy_easy_substitutive` | Some clean chunks replaced by easy distractors |
| `test_noisy_hard_substitutive` | Some clean chunks replaced by hard distractors |

### Why Additive and Substitutive Noise?

Additive noise tests robustness when distractors are appended to clean context. However, this can increase input length and cause truncation.

Substitutive noise replaces part of the clean context with distractors while keeping the number of chunks more comparable. This makes it a more reliable length-controlled diagnostic setting.

## Metrics

The evaluation package computes the following groups of metrics.

### Summarization Quality

- ROUGE-1
- ROUGE-2
- ROUGE-L
- ROUGE-Lsum
- BERTScore Precision / Recall / F1

### Retrieval Diagnostics

- Hit@K
- Precision@K
- Recall@K
- MRR
- nDCG@K
- noise chunk ratio

Retrieval relevance is approximated by **source-document membership**. This proxy checks whether retrieved chunks come from the target paper, but it does not guarantee that each chunk contains summary-relevant evidence.

### Truncation Diagnostics

- source token length
- target token length
- prediction token length
- source truncation risk
- target truncation risk
- model-specific truncation rate

Model-specific truncation is important because BART and T5 use different tokenizers.

### Robustness Metrics

- retention rate
- absolute degradation
- relative degradation
- additive vs. substitutive comparison

### Faithfulness Proxies

- source-supported entity rate
- unsupported entity rate
- number preservation
- source token support rate
- sentence support proxy

These are treated as lightweight faithfulness proxies, not as full factual consistency metrics.

### Statistical Tests

- paired t-test
- Wilcoxon signed-rank test
- Cohen's dz
- Cliff's delta
- rank-biserial correlation
- bootstrap 95% confidence interval
- Holm-Bonferroni correction

## Repository Structure

A recommended repository structure is:

```text
.
├── README.md
├── paper/
│   ├── main_evaluation_framework_final.tex
│   └── main_evaluation_framework_final.pdf
├── data_builder/
│   ├── databuildt_fixed_v2.py
│   ├── retrieval_tokenizer.py
│   ├── rulebase_chunkforpdf.py
│   └── summarized.py
├── training/
│   ├── train_bart_t5_runpod_v2.py
│   ├── run_train_bart_t5_auto_1epoch.sh
│   └── requirements_train_runpod_v2.txt
├── evaluation/
│   ├── eval_rankB_metrics_runpod.py
│   ├── run_rankB_metrics_full.sh
│   ├── run_rankB_metrics_from_predictions.sh
│   ├── run_rankB_metrics_rouge_only.sh
│   └── requirements_rankB_metrics.txt
├── outputs/
│   ├── rankB_compact_paper_table.csv
│   ├── rankB_metrics_summary.csv
│   ├── rankB_robustness_degradation.csv
│   ├── rankB_retrieval_summary.csv
│   ├── rankB_truncation_summary.csv
│   └── rankB_all_metrics_tables.xlsx
└── prepared_data_rankB_fixed_v2/
    ├── train_noiseaware.jsonl
    ├── train_clean_matched.jsonl
    ├── valid_noiseaware.jsonl
    ├── valid_clean_matched.jsonl
    ├── test_clean.jsonl
    ├── test_noisy_easy_additive.jsonl
    ├── test_noisy_hard_additive.jsonl
    ├── test_noisy_easy_substitutive.jsonl
    └── test_noisy_hard_substitutive.jsonl
```

Large datasets and model checkpoints should normally be excluded from Git and stored externally.

## Environment

The experiments were designed for a RunPod environment with an NVIDIA RTX 4090 24GB GPU.

Recommended Python environment:

```bash
python -m pip install --upgrade pip
pip install -r requirements_train_runpod_v2.txt
pip install -r requirements_rankB_metrics.txt
```

For BERTScore, PyTorch 2.6 or higher is recommended because recent Transformers versions restrict unsafe `torch.load` usage for older PyTorch versions.

## Dataset Construction

Example command:

```bash
python databuildt_fixed_v2.py   --arxiv_dir ./dataset/arxiv   --output_dir ./prepared_data_rankB_fixed_v2   --train_limit 20000   --valid_limit 1000   --test_limit 500   --min_target_words 30   --max_target_words 512   --final_k 3   --noise_k 2   --min_chunks 1   --noise_pool_limit 10000   --noise_pool_strategy heldout_train_tail   --test_noise_pool_offset 30000   --test_noise_pool_limit 10000   --substitutive_clean_k 1   --num_workers 2   --encode_batch_size 16   --paper_batch 100   --clean_control_mode unique   --seed 42   --laptop_safe
```

The generated test files should contain:

```text
test_clean.jsonl
test_noisy_easy_additive.jsonl
test_noisy_hard_additive.jsonl
test_noisy_easy_substitutive.jsonl
test_noisy_hard_substitutive.jsonl
```

Each test condition contains 500 samples in the full experiment.

## Training

The training setup uses the same training budget for clean-matched and noise-aware variants.

Example RunPod command:

```bash
cd /workspace

DATA_DIR=/workspace/prepared_data_rankB_fixed_v2 OUT_ROOT=/workspace/outputs/bart_t5_auto_1epoch ./run_train_bart_t5_auto_1epoch.sh
```

Expected output directories:

```text
/workspace/outputs/bart_t5_auto_1epoch/
  01_bart_base_noiseaware/
  02_bart_base_clean_matched/
  03_t5_base_noiseaware/
  04_t5_base_clean_matched/
```

## Evaluation

After training, run the full evaluation:

```bash
cd /workspace

python eval_rankB_metrics_runpod.py   --data_dir /workspace/prepared_data_rankB_fixed_v2   --out_root /workspace/outputs/bart_t5_auto_1epoch   --output_dir /workspace/eval_outputs/rankB_metrics_full   --skip_generation   --max_source_length 1024   --max_target_length 512   --generation_max_length 320   --bertscore_batch_size 16   --bertscore_model roberta-large   --bertscore_max_length 512   --length_tokenizer facebook/bart-base
```

If BERTScore is too slow, run:

```bash
python eval_rankB_metrics_runpod.py   --data_dir /workspace/prepared_data_rankB_fixed_v2   --out_root /workspace/outputs/bart_t5_auto_1epoch   --output_dir /workspace/eval_outputs/rankB_metrics_rouge_only   --skip_generation   --skip_bertscore   --max_source_length 1024   --max_target_length 512   --generation_max_length 320   --length_tokenizer facebook/bart-base
```

## Main Output Files

The evaluation script produces:

```text
rankB_metrics_record_level.csv
rankB_metrics_summary.csv
rankB_compact_paper_table.csv
rankB_paired_noiseaware_vs_clean.csv
rankB_robustness_degradation.csv
rankB_additive_vs_substitutive.csv
rankB_retrieval_summary.csv
rankB_truncation_summary.csv
rankB_all_metrics_tables.xlsx
```

Recommended files for reporting:

| File | Purpose |
|---|---|
| `rankB_compact_paper_table.csv` | compact paper-level performance table |
| `rankB_robustness_degradation.csv` | robustness retention/degradation |
| `rankB_retrieval_summary.csv` | retrieval diagnostics |
| `rankB_truncation_summary.csv` | truncation diagnostics |
| `rankB_paired_noiseaware_vs_clean.csv` | statistical comparison |
| `rankB_all_metrics_tables.xlsx` | complete workbook |

## Key Findings

The final paper frames the results as a case study of the proposed evaluation protocol.

Main findings:

- BART-base noise-aware training shows modest robustness gains under hard additive noise.
- Substitutive noise provides a more reliable length-controlled diagnostic than additive noise.
- T5-base additive-noise results are not used as primary robustness evidence because of high truncation.
- Small BERTScore gains should be interpreted conditionally and weighed against clean-condition degradation and training cost.
- Truncation diagnostics are essential for deciding whether a robustness claim is reliable.

## Practical Interpretation

This project does not claim that noise-aware training universally improves RAG summarization. Instead, it argues that robustness claims should be accepted only when:

1. retrieval noise is verified;
2. truncation is controlled or reported;
3. additive results are checked against substitutive results;
4. performance gains survive statistical testing;
5. practical utility is positive after accounting for clean-condition trade-offs.

## Citation

If you use this repository, please cite the paper or repository as:

```bibtex
@misc{truncation_aware_rag_eval,
  title        = {Truncation-Aware Evaluation Protocol for Noise-Aware RAG Summarization},
  author       = {Your Name},
  year         = {2026},
  note         = {Evaluation framework and experimental pipeline for noise-aware RAG summarization}
}
```

Please replace `Your Name` with the correct author information before public release.

## Limitations

- Retrieval relevance is approximated by source-document membership rather than human relevance labels.
- Faithfulness metrics are lightweight proxies and should not be interpreted as full factual consistency evaluation.
- Results are based on BART-base and T5-base; larger long-context models may reduce truncation but require a different compute budget.
- The study emphasizes evaluation validity rather than leaderboard-style performance optimization.

## License

- MIT License
- Apache License 2.0
- CC BY 4.0 for paper/materials

## Contact

For questions, issues, or reproduction details, please open a GitHub issue or contact the repository maintainer.
