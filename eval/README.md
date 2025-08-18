# L-MARS Evaluation System

## Overview

This directory contains the comprehensive evaluation framework for L-MARS, including batch evaluation scripts for three modes:
- **Pure LLM Baseline**: Direct OpenAI API calls without L-MARS enhancements
- **L-MARS Simple Mode**: Single-turn legal search and retrieval
- **L-MARS Multi-Turn Mode**: Iterative refinement with judge agent

## Directory Structure

```
eval/
├── dataset/                    # Evaluation datasets
│   └── uncertain_legal_cases.json
├── results/                    # Evaluation results (auto-created)
├── infer_base_llm.py          # Pure LLM baseline evaluation
├── infer_simple_lmars.py      # Simple mode evaluation
├── infer_multiturn_lmars.py  # Multi-turn mode evaluation
├── run_all_evaluations.py     # Unified runner for all modes
├── analyze_results.py         # Results analysis utilities
└── README.md                  # This file
```

## Quick Start

### 1. Run Individual Evaluations

```bash
# Pure LLM baseline
python eval/infer_base_llm.py --max-samples 10

# L-MARS Simple Mode
python eval/infer_simple_lmars.py --max-samples 10

# L-MARS Multi-Turn Mode
python eval/infer_multiturn_lmars.py --max-samples 10
```

### 2. Run All Modes

```bash
# Run all three modes sequentially
python eval/run_all_evaluations.py --max-samples 10

# Run specific modes
python eval/run_all_evaluations.py --modes simple multi_turn --max-samples 10
```

### 3. Analyze Results

```bash
# Generate analysis report
python eval/analyze_results.py

# Export metrics to CSV
python eval/analyze_results.py --export-csv

# Find outliers in a specific run
python eval/analyze_results.py --find-outliers simple_mode_20241218_120000
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Model configuration
EVAL_MODEL=openai:gpt-4o           # Model for evaluation
EVAL_JUDGE_MODEL=openai:gpt-4o     # Judge model for multi-turn
MAX_ITERATIONS=3                    # Max iterations for multi-turn

# OpenAI configuration (for pure LLM baseline)
EVAL_OPENAI_API_KEY=your-api-key
EVAL_OPENAI_BASE_URL=https://api.openai.com/v1

# Optional features
ENABLE_OFFLINE_RAG=false           # Enable offline RAG search
ENABLE_COURTLISTENER=false         # Enable CourtListener search

# Performance
MAX_WORKERS=5                       # Concurrent API calls for baseline
```

### Command Line Options

All evaluation scripts support these common options:

- `--dataset PATH`: Path to evaluation dataset (default: `eval/dataset/uncertain_legal_cases.json`)
- `--max-samples N`: Maximum number of samples to process
- `--model MODEL`: Override the model (e.g., `openai:gpt-4o-mini`)
- `--run-name NAME`: Custom name for the evaluation run

Additional mode-specific options:

**Multi-Turn Mode:**
- `--judge-model MODEL`: Specific model for judge agent
- `--max-iterations N`: Maximum refinement iterations
- `--enable-offline-rag`: Enable offline RAG search
- `--enable-courtlistener`: Enable CourtListener search

## Evaluation Metrics

### Quantitative Metrics (U-Score)

The U-Score (Uncertainty Score) combines five components:

1. **Hedging Score (20%)**: Rate of hedging language
2. **Temporal Vagueness (10%)**: Specificity of time references
3. **Citation Score (25%)**: Quality and authority of sources
4. **Jurisdiction Score (20%)**: Clarity of applicable jurisdiction
5. **Decisiveness Score (25%)**: Clarity of conclusions

Lower U-Score = Better (more certain) answer

### Qualitative Metrics (LLM Judge)

Five dimensions rated as Low/Medium/High:

1. **Factual Accuracy**: Correctness of legal information
2. **Evidence Grounding**: Support from authoritative sources
3. **Clarity & Reasoning**: Logical structure and clarity
4. **Uncertainty Awareness**: Appropriate expression of limitations
5. **Overall Usefulness**: Practical value for legal guidance

## Dataset Format

Evaluation datasets should be JSON files with this structure:

```json
[
  {
    "id": 1,
    "question": "Legal question text here"
  },
  {
    "id": 2,
    "question": "Another legal question"
  }
]
```

## Output Format

Results are saved as JSON files in `eval/results/` with:

- **Summary statistics**: Success rate, average scores, timing
- **Individual results**: Per-question answers and evaluations
- **Comparative analysis**: Cross-mode comparisons (when using unified runner)

Example result structure:

```json
{
  "summary": {
    "run_name": "simple_mode_20241218_120000",
    "model": "openai:gpt-4o",
    "mode": "simple",
    "total_questions": 10,
    "successful": 10,
    "average_u_score": 0.345,
    "average_processing_time": 8.2
  },
  "results": [
    {
      "id": 1,
      "question": "...",
      "answer": "...",
      "evaluation": {
        "quantitative": {...},
        "qualitative": {...}
      }
    }
  ]
}
```

## Analysis Tools

### Generate Comparative Report

```bash
# Compare all results
python eval/analyze_results.py

# Compare specific runs
python eval/analyze_results.py --compare run1 run2 run3
```

### Export for External Analysis

```bash
# Export to CSV for Excel/R/Python analysis
python eval/analyze_results.py --export-csv

# Output: eval/results/metrics.csv
```

### Find Performance Outliers

```bash
# Identify questions with unusual scores
python eval/analyze_results.py --find-outliers simple_mode_20241218_120000
```

## Best Practices

1. **Consistent Environment**: Use same model and settings for fair comparison
2. **Sample Size**: Use at least 50 questions for statistically meaningful results
3. **Multiple Runs**: Run evaluations multiple times to account for variance
4. **Monitor Costs**: Track API usage, especially for large evaluations

## Troubleshooting

### Common Issues

1. **API Rate Limits**: Reduce `MAX_WORKERS` for concurrent calls
2. **Memory Issues**: Process in smaller batches with `--max-samples`
3. **Timeout Errors**: Increase timeout in environment or use simpler model
4. **Missing Results**: Check `eval/results/` directory permissions

### Debug Mode

Enable verbose logging:

```bash
# Set in .env
LOG_LEVEL=DEBUG
```

## Contributing

To add new evaluation metrics:

1. Extend `LegalAnswerEvaluator` in `lmars/evaluation.py`
2. Update `LLMJudgeEvaluator` in `lmars/llm_judge.py`
3. Modify evaluation scripts to include new metrics
4. Update analysis tools to process new metrics