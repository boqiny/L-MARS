# L-MARS - Enhanced Multi-Agent Workflow for Legal QA

L-MARS (Legal Multi-Agent Framework for Orchestrated Reasoning and Agentic Search)combines multiple search strategies, agent-based reasoning, and evaluation systems for legal QA.

## Key Features

### Operating Mode
- **Simple Mode**: Fast single-turn pipeline with multiple search results
- **Multi-Turn Mode**: Iterative refinement with judge agent evaluation and deep content extraction

### Retrieval Source
1. **Online Search** (Default): Web search via Serper API - **Always enabled**
   - Simple mode: Quick search returning 5 results with snippets
   - Multi-turn mode: Deep search returning 3 results with full content extraction
2. **Local RAG** (Optional): BM25-based local document retrieval from `inputs/` folder - Enable with `--offline-rag`
3. **Case Law** (Optional): Legal case retrieval via CourtListener API - Enable with `--courtlistener`

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/l-mars.git
cd l-mars

# Install the package with dependencies
pip install -e .

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys:
# - OPENAI_API_KEY (required)
# - SERPER_API_KEY (required, for web search)
# - COURTLISTENER_API_TOKEN (optional, for case law)
```

### Setting Up Local Documents for Offline RAG

Place your legal documents in markdown format in the `inputs/` folder:

```bash
# Create inputs directory
mkdir -p inputs

# Add your markdown documents
cp your_legal_docs/*.md inputs/
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

### Streamlit Web Interface

```bash
# Run the web interface
streamlit run app/main.py
```

## System Architecture

### Simple Mode Pipeline
```
User Query
    ↓
Search Generation (based on enabled sources)
    ↓
├── Offline RAG (BM25, if enabled)
├── Serper Web Search (5 results, snippets only)
└── CourtListener (if enabled)
    ↓
Summary Agent (LLM)
    ↓
Dual Evaluation
├── Quantitative: U-Score (0-1)
└── Qualitative: LLM Judge (Low/Medium/High)
    ↓
Final Answer with Metrics
```

### Multi-Turn Mode Pipeline
```
User Query
    ↓
Query Agent: Generate Follow-up Questions
    ↓
[Optional: User Responses]
    ↓
Iterative Loop (max_iterations):
    ├── Generate Search Queries
    ├── Execute Deep Search (3 results with content)
    ├── Judge Agent Evaluation (temperature=0)
    │   ├── Chain-of-Thought Reasoning
    │   ├── Source Quality Check
    │   ├── Date/Jurisdiction/Contradiction Analysis
    │   └── Sufficiency Decision
    └── If Insufficient: Generate Refinement Queries
    ↓
Summary Agent → Dual Evaluation → Final Answer
```

## Evaluation System

### Quantitative U-Score (0-1 scale)
Weighted combination of five metrics:
- **Hedging Score (20%)**: Measures uncertainty language and conditional statements
- **Temporal Vagueness (10%)**: Detects vague temporal references
- **Citation Score (25%)**: Evaluates source citations and references
- **Jurisdiction Score (20%)**: Checks for jurisdiction clarity and specificity
- **Decisiveness Score (25%)**: Assesses answer clarity and directness

### Qualitative LLM Judge
Five assessment dimensions (Low/Medium/High):
- **Factual Accuracy**: Correctness of legal information
- **Evidence Grounding**: Quality of source support
- **Clarity & Reasoning**: Logical structure and explanation
- **Uncertainty Awareness**: Appropriate hedging and limitations
- **Overall Usefulness**: Practical value for the user

## Evaluation

The `eval/` folder contains tools for systematic evaluation:

```bash
# Evaluate all three modes on a dataset
cd eval
python run_all_evaluations.py

# Individual mode evaluation
python infer_base_llm.py          # Pure LLM baseline
python infer_simple_lmars.py      # Simple mode with search
python infer_multiturn_lmars.py   # Multi-turn with refinement

# Analyze results
python analyze_results.py
```

## Disclaimers

- This system provides legal information, not legal advice
- Always consult with qualified attorneys for specific legal matters
- Results are based on available sources and may not be comprehensive
- Laws vary by jurisdiction and change over time

## License

MIT License - See LICENSE file for details
