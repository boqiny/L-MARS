"""
Push data/legalsearchqa.json to boqiny/LegalSearchQA on HuggingFace Hub.

Flattens choices array into choice_A/B/C/D columns and maps gold index to letter,
so the dataset renders cleanly as a table on the HuggingFace data card.

Run: python3 push_to_hf.py
"""

import json
from pathlib import Path
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

DATA_PATH = "data/legalsearchqa.json"
REPO_ID = "boqiny/LegalSearchQA"
GOLD_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}

README = """\
---
license: cc-by-4.0
language:
- en
task_categories:
- question-answering
- multiple-choice
tags:
- law
- legal
- benchmark
- multiple-choice
- retrieval-augmented-generation
- rag-evaluation
pretty_name: LegalSearchQA
size_categories:
- n<1K
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
---

# LegalSearchQA

**LegalSearchQA** is a multiple-choice benchmark for evaluating legal knowledge retrieval and reasoning over current U.S. and international law. Each question requires finding and interpreting recent legal sources (statutes, regulations, court decisions, agency guidance) that may have changed after common LLM training cutoffs.

## Motivation

Many legal questions have answers that change over time — executive orders are revoked, tax thresholds adjust annually, regulations are amended. Standard LLM benchmarks do not capture this temporal sensitivity. LegalSearchQA is designed to test **retrieval-augmented** systems that must find and reason over live legal sources rather than rely on memorized training data.

## Dataset Summary

| Field | Value |
|---|---|
| Split | `test` |
| Examples | 50 |
| Question type | 4-choice multiple choice |
| Domains | 10 |
| Verified date | 2026-03-26 |

## Data Fields

| Column | Type | Description |
|---|---|---|
| `id` | string | Unique question identifier (`lsqa_001` … `lsqa_050`) |
| `question` | string | The legal question |
| `choice_A` | string | Answer choice A |
| `choice_B` | string | Answer choice B |
| `choice_C` | string | Answer choice C |
| `choice_D` | string | Answer choice D |
| `answer` | string | Correct answer letter (`A`/`B`/`C`/`D`) |
| `rationale` | string | Explanation of the correct answer with source citations |
| `domain` | string | Legal domain (see below) |
| `category` | string | Question category (see below) |
| `difficulty` | string | `easy` / `medium` / `hard` |
| `source_name` | string | Name of the authoritative source |
| `source_url` | string | URL of the authoritative source |
| `date_verified` | string | Date the answer was last verified (YYYY-MM-DD) |

### Domains

`corporate_law` · `criminal_law` · `drug_policy` · `immigration_law` · `labor_law` · `privacy_law` · `securities_regulation` · `state_law` · `tax_law` · `technology_regulation`

### Categories

| Category | Description |
|---|---|
| `status` | Current legal status of a law, order, or regulation |
| `threshold` | Numeric thresholds (dollar amounts, time limits, percentages) |
| `requirement_check` | Whether a specific requirement applies |
| `scope` | What entities or situations a law covers |
| `authority` | Which agency or court has jurisdiction |

## Usage

```python
from datasets import load_dataset

ds = load_dataset("boqiny/LegalSearchQA", split="test")
print(ds[0])
```

## Citation

If you use this dataset, please cite:

```
@misc{wang2025lmarslegalmultiagentworkflow,
  title         = {L-MARS: Legal Multi-Agent Workflow with Orchestrated Reasoning and Agentic Search},
  author        = {Ziqi Wang and Boqin Yuan},
  year          = {2025},
  eprint        = {2509.00761},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2509.00761}
}
```

## License

[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
"""

# ── Build dataset ─────────────────────────────────────────────────────────────

with open(DATA_PATH) as f:
    raw = json.load(f)

rows = []
for item in raw:
    choices = item["choices"]
    rows.append({
        "id":            item["id"],
        "question":      item["question"],
        "choice_A":      choices[0],
        "choice_B":      choices[1],
        "choice_C":      choices[2],
        "choice_D":      choices[3],
        "answer":        GOLD_MAP[item["gold"]],
        "rationale":     item["rationale"],
        "domain":        item["domain"],
        "category":      item["category"],
        "difficulty":    item["difficulty"],
        "source_name":   item["source_name"],
        "source_url":    item["source_url"],
        "date_verified": item["date_verified"],
    })

ds = Dataset.from_list(rows)
dataset_dict = DatasetDict({"test": ds})

print(f"Dataset: {len(ds)} examples")
print(f"Columns: {ds.column_names}")

# ── Push dataset ──────────────────────────────────────────────────────────────

dataset_dict.push_to_hub(
    REPO_ID,
    private=False,
    commit_message="Add LegalSearchQA benchmark dataset",
)

# ── Push dataset card (README.md) ─────────────────────────────────────────────

api = HfApi()
api.upload_file(
    path_or_fileobj=README.encode(),
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="dataset",
    commit_message="Add dataset card",
)

print(f"\nDone! View at: https://huggingface.co/datasets/{REPO_ID}")
