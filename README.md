# L-MARS - Legal Multi-Agent Reasoning System

L-MARS (Legal Multi-Agent Reasoning System) is an advanced legal question-answering framework that combines multiple search strategies, agent-based reasoning, and dual evaluation systems for comprehensive legal research.

## Key Features

### Dual Operating Modes
- **Simple Mode**: Fast single-turn pipeline with multiple search results
- **Multi-Turn Mode**: Iterative refinement with judge agent evaluation and deep content extraction

### Three Retrieval Sources (Configurable)
1. **Online Search** (Default): Web search via Serper API - **Always enabled**
   - Simple mode: Quick search returning 5 results with snippets
   - Multi-turn mode: Deep search returning 3 results with full content extraction
2. **Offline RAG** (Optional): BM25-based local document retrieval from `inputs/` folder - Enable with `--offline-rag`
3. **Case Law** (Optional): Legal case retrieval via CourtListener API - Enable with `--courtlistener`

### Dual Evaluation System
- **Quantitative U-Score**: Objective metrics measuring hedging, citations, jurisdiction clarity, temporal specificity, and decisiveness
- **Qualitative LLM Judge**: Subjective assessment of factual accuracy, evidence grounding, clarity, uncertainty awareness, and overall usefulness

## Operating Modes

### Simple Mode (Default)
- **Architecture**: Query → Search (5 results) → Summarize → Evaluate
- **Search Strategy**: Uses `search_serper_web` for fast retrieval of multiple results
- **Evaluation**: Combined quantitative (U-score) and qualitative (LLM judge) assessment
- **Best for**: Quick legal questions needing broad coverage
- **Temperature**: Variable (default model settings)

### Multi-Turn Mode
- **Architecture**: Query → Follow-up Questions → Iterative Search → Judge Evaluation → Refinement → Final Answer
- **Search Strategy**: Uses `search_serper_with_content` for deep content extraction (3 results per iteration)
- **Judge Agent**: Evaluates sufficiency with chain-of-thought reasoning (temperature=0 for reproducibility)
- **Refinement**: Automatically generates new search queries based on missing information
- **Max Iterations**: Configurable (default: 3)
- **Best for**: Complex legal questions requiring thorough, iterative research

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/l-mars.git
cd l-mars

# Install dependencies
pip install -r requirements.txt
pip install scikit-learn  # For offline RAG

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys:
# - OPENAI_API_KEY (required)
# - SERPER_API_KEY (required, for web search)
# - COURTLISTENER_API_TOKEN (optional, for case law)
```

### Setting Up Local Documents

Place your legal documents in markdown format in the `inputs/` folder:

```bash
# Create inputs directory
mkdir -p inputs

# Add your markdown documents
cp your_legal_docs/*.md inputs/
```

The system will automatically index and search these documents.

### Command Line Usage

#### Simple Mode (Default - Online Search Only)
```bash
# Quick legal research with online search only (default)
python main.py "Can an F1 student work remotely?"

# Enable offline RAG for local documents
python main.py --offline-rag "Can an F1 student work remotely?"

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

#### Interactive Mode
```bash
# Start interactive session
python main.py --interactive

# Commands:
# - Type 'simple' to switch to Simple Mode
# - Type 'multi' to switch to Multi-Turn Mode  
# - Type 'sources' to configure search sources
# - Type 'help' for available commands
# - Type your legal questions directly

# In interactive mode, configure sources:
[simple]> sources           # Show current configuration
[simple]> sources offline   # Toggle offline RAG
[simple]> sources court     # Toggle CourtListener
[simple]> sources all       # Enable all sources
[simple]> sources none      # Use web search only
```

### Python API

```python
from lmars import create_workflow

# Simple Mode with online search only (default)
workflow = create_workflow(mode="simple")
result = workflow.run("Can I form an LLC as a non-resident?")
print(result["final_answer"].answer)
print(f"U-Score: {result['evaluation_metrics'].u_score:.3f}")
print(f"Quality: {result['combined_evaluation']['qualitative'].overall_usefulness.level}")

# Simple Mode with all sources
workflow = create_workflow(
    mode="simple",
    enable_offline_rag=True,
    enable_courtlistener=True
)
result = workflow.run("Can I form an LLC as a non-resident?")

# Multi-Turn Mode with deep search
workflow = create_workflow(
    mode="multi_turn",
    max_iterations=3,
    judge_model="openai:gpt-4o",  # Explicit judge model with temperature=0
    enable_offline_rag=True
)
result = workflow.run("Complex IP licensing question...")

# Access evaluation results
print(f"Iterations: {result['iterations']}")
print(f"U-Score: {result['evaluation_metrics'].u_score:.3f}")
print(f"LLM Judge: {result['combined_evaluation']['qualitative'].summary}")
```

### Adding Documents to Offline RAG

```python
from lmars.tools.offline_rag_tool import add_document_to_rag

# Add a new document programmatically
content = """
# Legal Document Title

Content of your legal document in markdown format...
"""

add_document_to_rag(content, "new_document.md")
```

### Streamlit Web Interface

```bash
# Run the web interface
streamlit run app/main.py
```

The web interface provides:
- Mode selection (Simple vs Multi-Turn)
- Model configuration
- Interactive follow-up questions
- Visual results display
- Automatic search across all sources

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

## Project Structure

```
l-mars/
├── inputs/                         # Local legal documents (markdown)
│   ├── f1_student_employment.md
│   └── startup_legal_requirements.md
├── lmars/
│   ├── __init__.py                # Package exports
│   ├── workflow.py                # Core workflow orchestration
│   ├── agents.py                  # Agent implementations (no confidence scores)
│   ├── cli.py                     # Command-line interface
│   ├── evaluation.py              # Quantitative U-score evaluation
│   ├── llm_judge.py               # Qualitative LLM-as-judge evaluation
│   ├── trajectory_tracker.py      # Execution tracking
│   ├── result_logger.py           # Result logging
│   └── tools/                     # External API integrations
│       ├── offline_rag_tool.py    # BM25-based local search
│       ├── serper_search_tool.py   # Dual-mode web search
│       └── courtlistener_tool.py   # Case law search
├── eval/                           # Batch evaluation tools
│   ├── README.md                  # Evaluation documentation
│   ├── infer_base_llm.py          # Pure LLM baseline
│   ├── infer_simple_lmars.py      # Simple mode evaluation
│   ├── infer_multiturn_lmars.py   # Multi-turn evaluation
│   ├── run_all_evaluations.py     # Unified evaluation runner
│   └── analyze_results.py         # Result analysis utilities
├── app/
│   └── main.py                    # Streamlit web interface
├── test/
│   └── test_workflow.py           # Test suite
├── main.py                        # CLI entry point
└── requirements.txt               # Dependencies
```

## Configuration

### Environment Variables
```env
OPENAI_API_KEY=sk-...        # Required
SERPER_API_KEY=...           # Optional, for web search
COURTLISTENER_API_TOKEN=...  # Optional, for case law
ANTHROPIC_API_KEY=...        # Optional, for Claude models
```

### Workflow Configuration
```python
from lmars import WorkflowConfig, LMarsWorkflow

config = WorkflowConfig(
    mode="simple",              # or "multi_turn"
    llm_model="openai:gpt-4o",
    judge_model="openai:gpt-4o", # For multi-turn mode (temperature=0)
    max_iterations=3,           # For multi-turn mode
    enable_tracking=True,       # Enable trajectory tracking
    enable_offline_rag=False,   # Enable local document search
    enable_courtlistener=False  # Enable case law search
)

workflow = LMarsWorkflow(config)
result = workflow.run("Your legal question")

# Access evaluation metrics
print(f"U-Score: {result['evaluation_metrics'].u_score:.3f}")
print(f"Hedging: {result['evaluation_metrics'].hedging_score:.2f}")
print(f"Citations: {result['evaluation_metrics'].citation_score:.2f}")
print(f"Jurisdiction: {result['evaluation_metrics'].jurisdiction_score:.2f}")
print(f"Decisiveness: {result['evaluation_metrics'].decisiveness_score:.2f}")

# Access qualitative evaluation
if 'qualitative' in result['combined_evaluation']:
    qual = result['combined_evaluation']['qualitative']
    print(f"Factual Accuracy: {qual.factual_accuracy.level}")
    print(f"Evidence Grounding: {qual.evidence_grounding.level}")
    print(f"Overall Usefulness: {qual.overall_usefulness.level}")
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

## Offline RAG Features

- **BM25 Algorithm**: State-of-the-art ranking for document retrieval
- **Smart Chunking**: 500-char chunks with 100-char overlap
- **Automatic Indexing**: All markdown files in `inputs/` folder
- **Hot Reload**: Add documents anytime without restart

### Supported Document Format

Documents should be in markdown format with:
- Clear headings using `#`, `##`, etc.
- Structured content with lists and sections
- Metadata in frontmatter (optional)

Example structure:
```markdown
# Document Title

## Section 1
Content...

## Section 2
- Point 1
- Point 2

## References
...
```

## Batch Evaluation

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

Evaluation outputs include:
- Quantitative metrics (U-score components)
- Qualitative assessments (LLM judge)
- Response times and costs
- Comparative analysis across modes

## Testing

```bash
# Run test suite
python test/test_workflow.py

# Run with pytest
pytest test/

# Test search tools
python lmars/tools/serper_search_tool.py  # Test web search
python lmars/tools/offline_rag_tool.py    # Test local RAG
```

## API Reference

### Main Functions

#### `create_workflow(mode, llm_model, judge_model, max_iterations, enable_tracking, enable_offline_rag, enable_courtlistener)`
Creates a workflow instance with specified configuration.
- `mode`: "simple" or "multi_turn"
- `llm_model`: Main LLM model (e.g., "openai:gpt-4o")
- `judge_model`: Judge model for multi-turn (uses temperature=0)
- `max_iterations`: Maximum refinement iterations for multi-turn
- `enable_tracking`: Enable execution trajectory tracking
- `enable_offline_rag`: Enable local document search
- `enable_courtlistener`: Enable case law search

#### `workflow.run(query, user_responses)`
Executes the workflow with the given query.
- Returns dict with `final_answer`, `evaluation_metrics`, `combined_evaluation`

### Data Models (Pydantic BaseModels)

- `WorkflowConfig`: Configuration for workflow behavior
- `FollowUpQuestion`: Clarifying question with reason
- `QueryGeneration`: Search query with type and priority
- `SearchResult`: Structured result (source, title, content, url)
- `JudgmentResult`: Judge evaluation with chain-of-thought reasoning
- `FinalAnswer`: Structured answer with key points and sources
- `Person`: Person entity in legal context
- `QueryResult`: Detailed query analysis and categorization

**Note**: All confidence fields have been removed from the models for cleaner, more reliable outputs.

## Search Strategy Details

### Simple Mode Search
- Uses `search_serper_web()` for fast retrieval
- Returns 5 results with titles, URLs, and snippets
- No content extraction from individual pages
- Optimized for speed and broad coverage

### Multi-Turn Mode Search  
- Uses `search_serper_with_content()` for deep analysis
- Returns 3 results with full webpage content extraction
- Includes smart snippet matching and context extraction
- Judge agent evaluates with temperature=0 for reproducibility
- Automatic query refinement based on missing information

### Source Priority
1. **Offline RAG** (High Priority when enabled): BM25-based local search
2. **Web Search** (Always High Priority): Serper API with dual modes
3. **Case Law** (Medium Priority when enabled): CourtListener API

## Disclaimers

- This system provides legal information, not legal advice
- Always consult with qualified attorneys for specific legal matters
- Results are based on available sources and may not be comprehensive
- Laws vary by jurisdiction and change over time
- Local documents in `inputs/` should be kept up-to-date

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please read CONTRIBUTING.md for guidelines.

## Support

For issues, questions, or suggestions, please open an issue on GitHub.