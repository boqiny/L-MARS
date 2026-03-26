# L-MARS (Minimal Single-Turn)

L-MARS now runs a minimal single-turn legal QA pipeline on **LEXam only**.

Pipeline:

1. `QueryAgent` builds one structured query from the question.
2. `SearchAgent` retrieves evidence (live or cache snapshot).
3. `SummaryAgent` generates one answer + rationale.

Evaluation uses:

- Micro F1
- LLM-as-judge (offline evaluator only)

## Quickstart

```bash
pip install -r requirements.txt
```

## Dataset

Place your LEXam file at `data/lexam/lexam.jsonl` (or pass `--dataset-path`).
Expected schema per row:

```json
{"id":"...","input":"...","gold":"..."}
```

## Run L-MARS (cached)

```bash
python run/single_turn_pipeline.py \
  --dataset leexam \
  --model openai:gpt-4o-mini \
  --use-cache true \
  --output results/lmars_preds.jsonl

python eval/run_eval.py \
  --preds results/lmars_preds.jsonl \
  --judge-sample 20 \
  --llm_model openai:gpt-4o-mini
```

## Run Baseline (Qwen3)

```bash
python baselines/run_baseline.py \
  --model qwen3 \
  --dataset leexam \
  --use-cache true \
  --output results/qwen_preds.jsonl

python eval/run_eval.py \
  --preds results/qwen_preds.jsonl \
  --judge-sample 20 \
  --llm_model openai:gpt-4o-mini
```