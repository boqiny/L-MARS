# L-MARS

**L-MARS** stands for **Legal Multi-Agent Workflow with Orchestrated Reasoning and Agentic Search**.

📄 **Paper:** [L-MARS: Legal Multi-Agent Workflow with Orchestrated Reasoning and Agentic Search](https://arxiv.org/abs/2509.00761)

🤗 **Dataset:** [LegalSearchQA](https://huggingface.co/datasets/boqiny/LegalSearchQA)

L-MARS is a multi-agent legal question answering system designed for grounded answers over **current** legal information. It combines structured query decomposition, agentic web search, evidence filtering, and cited answer synthesis. The project also includes optional local retrieval over user-provided documents and CourtListener integration for case-law search.

![workflow](https://github.com/user-attachments/assets/c047aa4d-7d29-4a2d-bf32-e34a00d7058d)

## What L-MARS does

L-MARS supports two operating modes:

- **Simple Mode**: a single-pass retrieval pipeline that decomposes the question, searches for evidence, and synthesizes a grounded answer.
- **Multi-Turn Mode**: an iterative search-and-verify loop that refines queries until the evidence is sufficient or a maximum number of iterations is reached.

The system can use the following evidence sources:

- **Web search** via Serper
- **Local RAG** over user-provided documents using BM25
- **CourtListener** for case-law retrieval

## Pipeline overview

1. **Query Agent** parses the question into structured search intents.
2. **Search Agent** retrieves evidence from the enabled sources.
3. **Judge Agent** checks whether the evidence is sufficient and flags missing information.
4. **Summary Agent** writes the final answer with citations and rationale.

## Evaluation

The paper evaluates L-MARS on two settings:

- **LegalSearchQA**: a 50-question benchmark that requires post-training, time-sensitive legal knowledge.
- **Bar Exam QA**: a reasoning-focused benchmark where retrieval provides only limited gains.

Reported metrics in the paper focus on **accuracy**. The benchmark is designed for grounded legal QA rather than classification metrics such as micro F1.

## Installation

```bash
pip install -r requirements.txt
```

## Run L-MARS

### Simple Mode

Quick legal research with online search only:

```bash
python main.py "Your legal question"
```

Enable offline RAG for local documents:

```bash
python main.py --offline-rag "Your legal question"
```

Enable all sources (offline RAG + CourtListener + web search):

```bash
python main.py --all-sources "Your legal question"
```

Verbose output:

```bash
python main.py -v "Your legal question"
```

### Multi-Turn Mode

Run iterative research with refinement:

```bash
python main.py --multi "Complex contract dispute..."
```

Set a custom number of iterations:

```bash
python main.py --multi --max-iterations 5 "Your question"
```

## Benchmark scripts

If you are reproducing the paper's evaluation pipeline:

```bash
python run/single_turn_pipeline.py \
  --dataset legalsearchqa \
  --model openai:gpt-4o-mini \
  --use-cache true \
  --output results/lmars_preds.jsonl

python eval/run_eval.py \
  --preds results/lmars_preds.jsonl \
  --judge-sample 20 \
  --llm_model openai:gpt-4o-mini
```

## Citation

If you use L-MARS in your research, please cite:

```bibtex
@misc{wang2025lmarslegalmultiagentworkflow,
  title={L-MARS: Legal Multi-Agent Workflow with Orchestrated Reasoning and Agentic Search},
  author={Ziqi Wang and Boqin Yuan},
  year={2025},
  eprint={2509.00761},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2509.00761},
}
```
