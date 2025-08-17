# L-MARS - Enhanced Multi-Agent Workflow for Legal QA

L-MARS (Legal Multi-Agent Framework for Orchestrated Reasoning and Agentic Search) integrates diverse retrieval sources with structured outputs and multi-turn model reasoning for comprehensive legal research.

## Key Features

### Three Retrieval Sources (Configurable)
1. **Online Search** (Default): Web search via Serper API for up-to-date information - **Always enabled**
2. **Offline RAG** (Optional): Local legal document retrieval from the `inputs/` folder - Enable with `--offline-rag`
3. **Case Law** (Optional): Legal case retrieval through CourtListener API - Enable with `--courtlistener`

### Two Operating Modes

#### 1. Simple Mode (Default)
- **Single-turn pipeline** for quick legal research
- Searches all three sources simultaneously
- Uses reasoning model to integrate and provide answers
- **Best for**: Straightforward legal questions
- **Speed**: Fast response time

#### 2. Multi-Turn Mode
- **Query Agent** refines user queries through clarifying sub-questions
- **Judge Agent** evaluates sufficiency and relevance of retrieved evidence
- **Summary Agent** synthesizes the final answer
- Iterative loop adaptively searches and refines until evidence threshold is met
- **Best for**: Complex legal questions requiring thorough research

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

# Simple Mode with all sources
workflow = create_workflow(
    mode="simple",
    enable_offline_rag=True,
    enable_courtlistener=True
)
result = workflow.run("Can I form an LLC as a non-resident?")

# Multi-Turn Mode with selected sources
workflow = create_workflow(
    mode="multi_turn",
    max_iterations=3,
    enable_offline_rag=True  # Only add offline RAG
)
result = workflow.run("Complex IP licensing question...")

# Handle follow-up questions if needed
if result.get("needs_user_input"):
    questions = result["follow_up_questions"]
    # Collect user responses...
    result = workflow.run(query, user_responses)
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

## Architecture

### Simple Mode Flow
```
User Query → 
├── Offline RAG (Local Documents)
├── Serper API (Web Search)
└── CourtListener (Case Law)
    ↓
Summarize → Answer
```

### Multi-Turn Mode Flow
```
User Query → Follow-up Questions → User Input → 
Generate Queries → 
├── Offline RAG
├── Online Search
└── Case Law
    ↓
Judge Evaluation →
[If insufficient: Refine & Search again] → Final Summary
```

## Project Structure

```
l-mars/
├── inputs/                # Local legal documents (markdown)
│   ├── f1_student_employment.md
│   └── startup_legal_requirements.md
├── lmars/
│   ├── __init__.py       # Package exports
│   ├── workflow.py       # Core workflow orchestration
│   ├── agents.py         # Agent implementations
│   ├── cli.py           # Command-line interface
│   └── tools/           # External API integrations
│       ├── offline_rag_tool.py    # Local document search
│       ├── serper_search_tool.py   # Web search
│       └── courtlistener_tool.py   # Case law search
├── app/
│   └── main.py          # Streamlit web interface
├── test/
│   └── test_workflow.py # Test suite
├── main.py              # CLI entry point
└── requirements.txt     # Dependencies
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
    judge_model="openai:gpt-4o", # For multi-turn mode
    max_iterations=3,           # For multi-turn mode
    enable_tracking=True        # Enable trajectory tracking
)

workflow = LMarsWorkflow(config)
```

## Offline RAG Features

The offline RAG system provides:
- **BM25 Algorithm** - State-of-the-art ranking function for document retrieval
- **Smart Chunking** - Documents split into semantic chunks with overlap for better context
- **Automatic indexing** of markdown files in `inputs/` folder
- **Relevance scoring** with BM25 scores to rank results
- **No internet required** for local document search
- **Hot reload** - add documents anytime

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

## Testing

```bash
# Run test suite
python test/test_workflow.py

# Run with pytest (if installed)
pytest test/

# Test with sample documents
python test_simple_mode.py
```

## API Reference

### Main Functions

#### `create_workflow(mode, llm_model, judge_model, max_iterations, enable_tracking)`
Creates a workflow instance with specified configuration.

#### `workflow.run(query, user_responses)`
Executes the workflow with the given query and optional user responses.

### Data Models

- `WorkflowConfig`: Configuration for workflow behavior
- `FollowUpQuestion`: Clarifying question with reason
- `QueryGeneration`: Search query with type (offline_rag, web_search, case_law) and priority
- `SearchResult`: Structured search result from any source
- `JudgmentResult`: Judge evaluation of results
- `FinalAnswer`: Final structured answer with sources

## Search Sources Priority

1. **Offline RAG (BM25)** (High Priority): Searches local trusted documents using BM25 algorithm
   - Semantic chunking with 500 char chunks and 100 char overlap
   - Stop word removal and tokenization
   - BM25 scoring for optimal relevance ranking
2. **Web Search** (High Priority): Current information from the internet via Serper API
3. **Case Law** (Medium Priority): Legal precedents and cases via CourtListener API

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