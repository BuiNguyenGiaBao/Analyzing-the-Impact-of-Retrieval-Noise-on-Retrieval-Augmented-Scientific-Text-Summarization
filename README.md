# Evaluating Noise-Aware Training for Robust RAG-Based Scientific Document Summarization under Controlled Retrieval Noise

This repository contains the code, configuration files, and experimental outputs for the paper:

**“Evaluating Noise-Aware Training for Robust RAG-Based Scientific Document Summarization under Controlled Retrieval Noise”**

The project investigates how retrieval noise affects retrieval-augmented generation (RAG)-based scientific document summarization and whether noise-aware training can improve robustness under controlled noisy retrieval conditions.

## Overview

Retrieval-augmented generation is useful for summarizing long scientific documents because full papers often exceed the input length of standard encoder–decoder models. However, RAG performance depends strongly on retrieval quality. When the retrieved context contains irrelevant or source-incorrect chunks, the summarization model may receive distracting evidence and generate lower-quality summaries.

This project evaluates summary-quality robustness under three retrieval conditions:

1. **Clean retrieval**
   The model receives only target-document chunks.

2. **High-similarity noise**
   The model receives the same clean target-document chunks plus cross-document noise chunks that are highly similar to the task-level summarization query.

3. **Low-similarity noise**
   The model receives the same clean target-document chunks plus cross-document noise chunks that are distant from the task-level summarization query.

The study compares clean-trained and noise-aware variants of **T5-base**, **BART-base**, and **PEGASUS-arXiv**, with **LED-base** used as a clean-trained long-context baseline.

## Main Contributions

This repository supports three main contributions:

* A controlled query-conditioned retrieval-noise setting for evaluating RAG summarization robustness.
* A context-construction strategy that keeps the number of clean target-document chunks fixed while adding source-incorrect noise chunks.
* A comparative evaluation of noise-aware training across multiple summarization architectures.

## Experimental Design

The experiment is based on a local Arrow-formatted article–abstract dataset derived from the Cornell-University arXiv dataset on Kaggle.

Each sample contains:

* `article`: source document used for chunking and retrieval
* `abstract`: reference summary used for training and evaluation

The abstract is not used to construct the retrieval query or select retrieved chunks.

### Context Construction

The main context construction setting is:

| Condition             | Target Chunks | Noise Chunks | Total Chunks | Target Ratio | Noise Ratio |
| --------------------- | ------------: | -----------: | -----------: | -----------: | ----------: |
| Clean                 |             3 |            0 |            3 |         1.00 |        0.00 |
| High-similarity noise |             3 |            2 |            5 |         0.60 |        0.40 |
| Low-similarity noise  |             3 |            2 |            5 |         0.60 |        0.40 |

The noisy conditions add cross-document distractor chunks without removing clean target-document evidence. This allows the evaluation to measure the effect of added retrieval noise rather than the effect of missing evidence.

## Method Pipeline

The experimental pipeline consists of the following stages:

```text
arXiv article–abstract dataset
        |
        v
Document preprocessing
        |
        v
Section-aware chunking
        |
        v
Dense retrieval
        |
        v
MMR reranking
        |
        v
Controlled noise injection
        |
        v
Input construction
        |
        v
Model fine-tuning
        |
        v
Evaluation under clean and noisy retrieval conditions
```

## Models

The following models are evaluated:

| Model         | Training Setting              |
| ------------- | ----------------------------- |
| T5-base       | Clean-trained and noise-aware |
| BART-base     | Clean-trained and noise-aware |
| PEGASUS-arXiv | Clean-trained and noise-aware |
| LED-base      | Clean-trained baseline only   |

## Training Configuration

| Configuration        |     T5-base |   BART-base | PEGASUS-arXiv |    LED-base |
| -------------------- | ----------: | ----------: | ------------: | ----------: |
| Learning rate        |        3e-5 |        3e-5 |          2e-5 |        1e-5 |
| Effective batch size |           8 |           8 |             8 |           8 |
| Clean-trained epochs |           3 |           3 |             3 |           3 |
| Noise-aware epochs   |           1 |           1 |             1 |           — |
| Max input length     |        1024 |        1024 |          1024 |        2048 |
| Max output length    |         256 |         256 |           256 |         192 |
| Decoding strategy    | Beam search | Beam search |   Beam search | Beam search |
| Beam size            |           2 |           2 |             2 |           2 |
| Random seed          |          42 |          42 |            42 |          42 |

The clean-trained setting uses 20,000 clean training examples for 3 epochs.
The noise-aware setting uses 60,000 mixed examples for 1 epoch.
This keeps the number of training sample-passes comparable across settings.

## Evaluation Metrics

The models are evaluated using:

* ROUGE-1
* ROUGE-2
* ROUGE-L
* BERTScore F1
* Absolute degradation
* Relative degradation
* Robustness gain
* Paired bootstrap confidence intervals
* Paired t-tests

The evaluation focuses on **summary-quality robustness**. It does not directly evaluate factual consistency or evidence faithfulness.

## Main Results

### Overall Summarization Performance

| Model         | Training      | Condition |    R-1 |    R-2 |    R-L | BERT-F1 |
| ------------- | ------------- | --------- | -----: | -----: | -----: | ------: |
| T5-base       | Clean-trained | Clean     | 0.2587 | 0.0679 | 0.1678 |  0.8297 |
| T5-base       | Clean-trained | High-sim. | 0.2379 | 0.0563 | 0.1579 |  0.8250 |
| T5-base       | Clean-trained | Low-sim.  | 0.2345 | 0.0551 | 0.1563 |  0.8248 |
| T5-base       | Noise-aware   | Clean     | 0.3134 | 0.0797 | 0.1877 |  0.8335 |
| T5-base       | Noise-aware   | High-sim. | 0.2908 | 0.0660 | 0.1770 |  0.8289 |
| T5-base       | Noise-aware   | Low-sim.  | 0.2895 | 0.0666 | 0.1769 |  0.8291 |
| BART-base     | Clean-trained | Clean     | 0.3280 | 0.0871 | 0.1973 |  0.8429 |
| BART-base     | Clean-trained | High-sim. | 0.3018 | 0.0722 | 0.1849 |  0.8379 |
| BART-base     | Clean-trained | Low-sim.  | 0.2849 | 0.0658 | 0.1774 |  0.8358 |
| BART-base     | Noise-aware   | Clean     | 0.3174 | 0.0844 | 0.1927 |  0.8418 |
| BART-base     | Noise-aware   | High-sim. | 0.2969 | 0.0734 | 0.1840 |  0.8380 |
| BART-base     | Noise-aware   | Low-sim.  | 0.2937 | 0.0730 | 0.1820 |  0.8379 |
| PEGASUS-arXiv | Clean-trained | Clean     | 0.3531 | 0.0991 | 0.2065 |  0.8404 |
| PEGASUS-arXiv | Clean-trained | High-sim. | 0.3237 | 0.0796 | 0.1924 |  0.8343 |
| PEGASUS-arXiv | Clean-trained | Low-sim.  | 0.3104 | 0.0740 | 0.1865 |  0.8322 |
| PEGASUS-arXiv | Noise-aware   | Clean     | 0.3488 | 0.0971 | 0.2056 |  0.8397 |
| PEGASUS-arXiv | Noise-aware   | High-sim. | 0.3270 | 0.0845 | 0.1952 |  0.8355 |
| PEGASUS-arXiv | Noise-aware   | Low-sim.  | 0.3255 | 0.0842 | 0.1939 |  0.8353 |
| LED-base      | Clean-trained | Clean     | 0.3051 | 0.0819 | 0.1896 |  0.8415 |
| LED-base      | Clean-trained | High-sim. | 0.2784 | 0.0659 | 0.1761 |  0.8355 |
| LED-base      | Clean-trained | Low-sim.  | 0.2663 | 0.0622 | 0.1711 |  0.8341 |

### Paired Robustness Gain for BERTScore F1

| Model         | Condition |     ΔF1 | 95% CI             | p-value |
| ------------- | --------- | ------: | ------------------ | ------: |
| T5-base       | Clean     |  0.0038 | [0.0033, 0.0042]   |  < .001 |
| T5-base       | High-sim. |  0.0039 | [0.0033, 0.0044]   |  < .001 |
| T5-base       | Low-sim.  |  0.0042 | [0.0037, 0.0048]   |  < .001 |
| BART-base     | Clean     | -0.0012 | [-0.0016, -0.0007] |  < .001 |
| BART-base     | High-sim. |  0.0001 | [-0.0004, 0.0007]  |    .764 |
| BART-base     | Low-sim.  |  0.0021 | [0.0015, 0.0027]   |  < .001 |
| PEGASUS-arXiv | Clean     | -0.0007 | [-0.0012, -0.0003] |  < .001 |
| PEGASUS-arXiv | High-sim. |  0.0013 | [0.0007, 0.0018]   |  < .001 |
| PEGASUS-arXiv | Low-sim.  |  0.0031 | [0.0026, 0.0037]   |  < .001 |

## Key Findings

The results show that retrieval noise reduces summarization performance across clean-trained models. Noise-aware training improves robustness in a model-dependent manner:

* **T5-base** shows the most consistent gains across clean and noisy conditions.
* **BART-base** mainly benefits under low-similarity noise and shows a clean-performance trade-off.
* **PEGASUS-arXiv** shows a small clean-condition loss but gains robustness under noisy retrieval, especially under low-similarity noise.
* **LED-base** serves as a useful long-context baseline, but it also declines under noisy retrieval conditions.

Overall, the findings suggest that noise-aware training can improve summary-quality robustness under controlled retrieval noise, but its effect is not uniform across architectures and retrieval conditions.

## Dataset

The raw dataset source is:

**Cornell-University arXiv Dataset**
Kaggle: https://www.kaggle.com/datasets/Cornell-University/arxiv

The full raw dataset is not redistributed in this repository. Users should download the dataset directly from Kaggle and reproduce the local Arrow and JSONL files using the provided scripts.

## Repository Structure

A suggested repository structure is shown below:

```text
.
├── README.md
├── configs/
│   ├── t5_base.yaml
│   ├── bart_base.yaml
│   ├── pegasus_arxiv.yaml
│   └── led_base.yaml
├── data/
│   └── README.md
├── scripts/
│   ├── build_arrow_dataset.py
│   ├── chunk_articles.py
│   ├── build_retrieval_index.py
│   ├── build_noise_conditions.py
│   ├── train_model.py
│   └── evaluate_model.py
├── results/
│   ├── overall_results.csv
│   ├── robustness_gain.csv
│   └── metrics_by_condition.csv
├── paper/
│   └── main_overleaf_free_github_link.tex
└── requirements.txt
```

Adjust the filenames above if your local implementation uses different script names.

## Installation

Create a Python environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

A typical environment may include:

```text
torch
transformers
datasets
sentence-transformers
faiss-cpu
numpy
pandas
scikit-learn
rouge-score
bert-score
tqdm
```

If GPU acceleration is used, install the appropriate PyTorch version for your CUDA environment.

## Reproducibility Settings

Main experimental configuration:

```text
final_k = 3
noise_k = 2
min_chunks = 3
random_seed = 42
clean_train_examples = 20000
clean_validation_examples = 2000
noise_aware_train_examples = 60000
test_examples_per_condition = 1999
```

Noise construction:

```text
Clean condition:
3 target-document chunks

High-similarity noise condition:
3 target-document chunks + 2 high-similarity cross-document noise chunks

Low-similarity noise condition:
3 target-document chunks + 2 low-similarity cross-document noise chunks
```

## Example Workflow

### 1. Prepare the dataset

```bash
python scripts/build_arrow_dataset.py \
  --input_path data/raw/arxiv \
  --output_path data/processed/arxiv_arrow
```

### 2. Chunk articles

```bash
python scripts/chunk_articles.py \
  --input_path data/processed/arxiv_arrow \
  --output_path data/chunks \
  --chunk_size 150 \
  --chunk_overlap 30 \
  --min_chunk_words 40
```

### 3. Build retrieval and noise conditions

```bash
python scripts/build_noise_conditions.py \
  --chunks_path data/chunks \
  --output_path data/noise_conditions \
  --final_k 3 \
  --noise_k 2 \
  --min_chunks 3 \
  --seed 42
```

### 4. Train models

```bash
python scripts/train_model.py \
  --config configs/t5_base.yaml \
  --training_mode clean
```

```bash
python scripts/train_model.py \
  --config configs/t5_base.yaml \
  --training_mode noise_aware
```

### 5. Evaluate models

```bash
python scripts/evaluate_model.py \
  --model_path checkpoints/t5_base_noise_aware \
  --test_path data/noise_conditions/test \
  --output_path results/t5_base_noise_aware_results.csv
```

## Limitations

This project evaluates summary-quality robustness using automatic metrics such as ROUGE and BERTScore. These metrics compare generated summaries with reference abstracts, but they do not directly verify whether each generated statement is supported by the retrieved evidence.

Therefore, the results should be interpreted as evidence of **summary-quality robustness**, not as direct evidence of factual correctness or evidence faithfulness.

Future work should evaluate:

* factual consistency,
* evidence support,
* human evaluation,
* split-specific test noise pools,
* token-length and truncation effects,
* length-controlled noise baselines,
* alternative retrievers,
* larger summarization models,
* different noise ratios.

## Citation

If you use this repository, please cite the paper:

```bibtex
@inproceedings{buinguyen2026noiseaware,
  title     = {Evaluating Noise-Aware Training for Robust RAG-Based Scientific Document Summarization under Controlled Retrieval Noise},
  author    = {Bui Nguyen Gia Bao},
  year      = {2026},
  booktitle = {Conference Workshop Paper},
  note      = {Repository for controlled retrieval-noise evaluation in RAG-based scientific summarization}
}
```

Please update the venue name, page numbers, DOI, and publication details after acceptance.

## Author

**Bui Nguyen Gia Bao**
HUTECH University
Ho Chi Minh City, Vietnam
Email: [giabao.dl2005@gmail.com](mailto:giabao.dl2005@gmail.com)

## License

This repository is released for academic and research purposes. Please check the license file for details.

## Acknowledgment

This project uses publicly available scientific article–abstract data derived from the Cornell-University arXiv dataset on Kaggle and builds controlled retrieval-noise conditions for robust RAG-based scientific summarization evaluation.
