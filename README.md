# L-MARS - Enhanced Multi-Agent Workflow for Legal QA

📄 **Paper:** [L-MARS: Legal Multi-Agent Workflow with Orchestrated Reasoning and Agentic Search](https://arxiv.org/abs/2509.00761)

L-MARS (Legal Multi-Agent Framework for Orchestrated Reasoning and Agentic Search) combines multiple search strategies, agent-based reasoning, and evaluation systems for legal QA.
![workflow](https://github.com/user-attachments/assets/c047aa4d-7d29-4a2d-bf32-e34a00d7058d)

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
