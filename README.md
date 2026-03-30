# L-MARS (Minimal Single-Turn)

📄 **Paper:** [L-MARS: Legal Multi-Agent Workflow with Orchestrated Reasoning and Agentic Search](https://arxiv.org/abs/2509.00761)

🤗 The complete dataset with sources and rationales is available at [boqiny/LegalSearchQA](https://huggingface.co/datasets/boqiny/LegalSearchQA).

L-MARS (Legal Multi-Agent Framework for Orchestrated Reasoning and Agentic Search) combines multiple search strategies, agent-based reasoning, and evaluation systems for legal QA.
![workflow](https://github.com/user-attachments/assets/c047aa4d-7d29-4a2d-bf32-e34a00d7058d)

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

## Run L-MARS

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

The system will automatically index and search these documents.

### Command Line Usage

#### Simple Mode
```bash
# Quick legal research with online search only (default)
python main.py "Your legal question"

# Enable offline RAG for local documents
python main.py --offline-rag "Your legal question"

# Enable all sources (offline RAG + CourtListener + web search)
python main.py --all-sources "Your legal question"

# With verbose output
python main.py -v "Your legal question"
```

#### Multi-Turn Mode
```bash
# Thorough research with refinement
python main.py --multi "Complex contract dispute..."

# With custom iterations
python main.py --multi --max-iterations 5 "Your question"
```

## Citation

If you use **L-MARS** in your research, please cite:

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
