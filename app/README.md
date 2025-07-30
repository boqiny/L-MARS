# L-MARS Streamlit Frontend

Interactive web interface for the L-MARS Legal Research System built with Streamlit.

## 🌟 Features

### Core Functionality
- **Interactive Legal Research** - Submit legal questions and get comprehensive answers
- **Follow-up Question Handling** - Dynamic clarification questions for better context
- **Real-time Progress Tracking** - Watch the multi-agent workflow in action
- **Trajectory Analysis** - Analyze past research sessions with detailed insights
- **Model Configuration** - Customize LLM models for different components

### User Interface
- **Clean, Professional Design** - Intuitive interface designed for legal professionals
- **Real-time Updates** - Live progress tracking during research execution
- **Interactive Charts** - Visual representation of workflow progress and results
- **Responsive Layout** - Works on desktop, tablet, and mobile devices
- **Dark/Light Theme Support** - Automatic theme detection

### Advanced Features
- **Progressive Result Saving** - Results saved incrementally during execution
- **Confidence Scoring** - Visual confidence meters for research quality
- **Export Capabilities** - Export results to JSON, PDF, and CSV formats
- **System Health Monitoring** - Real-time system performance metrics
- **Error Recovery** - Graceful handling of model failures and timeouts

## 🚀 Quick Start

### Prerequisites
```bash
# Install required dependencies
pip install streamlit plotly pandas

# Set up API keys (at least one required)
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"  # Optional
```

### Launch the Application
```bash
# From project root
python run_streamlit.py

# Or directly with streamlit
streamlit run app/main.py
```

The application will open automatically in your browser at `http://localhost:8501`

## 📖 Usage Guide

### 1. Initial Setup
1. **System Initialization**: The system initializes automatically on first load
2. **Model Configuration**: Configure your preferred models in the sidebar
3. **API Key Validation**: Ensure your API keys are properly set

### 2. Asking Legal Questions
1. **Enter Your Question**: Use the main text area to input your legal query
2. **Start Research**: Click "Start Research" to begin the multi-agent workflow
3. **Follow-up Questions**: Answer any clarifying questions that appear
4. **View Results**: Review the comprehensive legal research results

### 3. Monitoring Progress
- **Real-time Steps**: Watch each workflow step execute in real-time
- **Search Results**: See search results as they're discovered
- **Judge Evaluations**: Monitor the judge agent's assessment of information sufficiency
- **Progress Charts**: Visual timeline of execution steps

### 4. Analyzing Results
- **Final Answer**: Comprehensive legal guidance with key points
- **Sources**: All sources used in the research
- **Confidence Score**: Visual confidence meter for answer quality
- **Disclaimers**: Important legal disclaimers and limitations

### 5. Trajectory Analysis
- **Past Runs**: Browse and analyze previous research sessions
- **Detailed Steps**: Examine each step of the workflow execution
- **Model Interactions**: View all model calls and responses
- **Performance Metrics**: Analyze timing and efficiency data

## ⚙️ Configuration Options

### Model Settings
- **Main LLM Model**: Primary model for query processing and summarization
- **Judge Model**: Specialized model for evaluating research sufficiency
- **Max Iterations**: Maximum search iterations before completing

### Available Models
- **OpenAI**: GPT-4o, GPT-4, GPT-3.5-turbo, O3-mini
- **Anthropic**: Claude 3 Opus, Claude 3 Sonnet

### System Settings
- **Theme**: Light/Dark mode (auto-detected)
- **Layout**: Wide/Centered layout options
- **Export Format**: JSON, PDF, CSV options

## 🛠️ Architecture

### Frontend Components
```
app/
├── main.py              # Main Streamlit application
├── components.py        # Reusable UI components
├── config.py           # Configuration settings
├── utils.py            # Utility functions
└── requirements.txt    # Dependencies
```

### Key Classes
- **LMarsStreamlitApp**: Main application class
- **ProgressTracker**: Real-time progress tracking
- **StreamlitLogger**: Application logging system
- **AppConfig**: Configuration management

### State Management
- **Session State**: Persistent state across page reloads
- **Workflow State**: Track current research progress
- **Configuration State**: User settings and preferences

## 🎨 User Interface

### Layout Structure
```
┌─ Header ─────────────────────────────────────┐
│ 🏛️ L-MARS Legal Research System              │
│ Status: Ready | Run ID: abc123...            │
└──────────────────────────────────────────────┘

┌─ Sidebar ────┐  ┌─ Main Content ─────────────┐
│ ⚙️ Config     │  │ 💬 Query Input            │
│ - Models      │  │ ❓ Follow-up Questions     │
│ - Settings    │  │ ⚡ Workflow Progress       │
│ - Status      │  │ 📋 Final Results          │
│ - Health      │  │ 📊 Trajectory Analysis    │
└───────────────┘  └───────────────────────────┘
```

### Color Scheme
- **Primary**: Blue (#1f77b4) - Actions and highlights
- **Success**: Green (#28a745) - Completed states
- **Warning**: Yellow (#ffc107) - Attention needed
- **Error**: Red (#dc3545) - Error states
- **Info**: Light Blue (#17a2b8) - Information

## 🔧 Development

### Adding New Features
1. Create new components in `components.py`
2. Add configuration options in `config.py`
3. Implement utility functions in `utils.py`
4. Update main application in `main.py`

### Custom Styling
- Modify `CUSTOM_CSS` in `config.py`
- Use Streamlit's built-in theming system
- Add custom HTML/CSS with `st.markdown(unsafe_allow_html=True)`

### Testing
```bash
# Run the application locally
python run_streamlit.py

# Test with different configurations
OPENAI_API_KEY=test streamlit run app/main.py
```

## 📊 Monitoring & Analytics

### Built-in Metrics
- **Response Times**: Track workflow execution speed
- **Model Usage**: Monitor API calls and token consumption
- **User Interactions**: Track query patterns and success rates
- **System Health**: Memory usage, session size, error rates

### Logging
- **Application Logs**: Track user actions and system events
- **Error Logging**: Comprehensive error tracking and reporting
- **Performance Logs**: Execution timing and resource usage

## 🔒 Security & Privacy

### Data Handling
- **No Persistent Storage**: Session data cleared on browser close
- **API Key Security**: Keys stored only in environment variables
- **Privacy Protection**: No user data transmitted to external services

### Legal Compliance
- **Disclaimer Integration**: Automatic legal disclaimers on all results
- **Data Retention**: Configurable trajectory data retention policies
- **Audit Trail**: Complete logging of all research activities

## 📚 Example Queries

Try these sample legal questions to explore the system:

1. **Employment Law**: "Can an F1 student work remotely for a US company while studying?"
2. **Business Law**: "What are the legal requirements for starting an LLC in California?"
3. **Privacy Law**: "Is it legal to record a conversation without consent in New York?"
4. **Intellectual Property**: "What constitutes fair use in trademark law?"
5. **Contract Law**: "Are non-compete agreements enforceable after termination?"

## 🆘 Troubleshooting

### Common Issues
- **System Initialization Failed**: Check API keys and internet connection
- **Slow Performance**: Try reducing max iterations or switching models
- **Empty Results**: Ensure API keys are valid and models are accessible
- **UI Not Responsive**: Clear browser cache and refresh the page

### Getting Help
- Check the application logs in the sidebar
- Monitor system health indicators
- Review trajectory data for debugging
- Contact support with specific error messages

## 🚀 Deployment

### Local Development
```bash
python run_streamlit.py
```

### Production Deployment
```bash
# Using Docker
docker build -t lmars-streamlit .
docker run -p 8501:8501 lmars-streamlit

# Using cloud platforms
# Deploy to Streamlit Cloud, Heroku, or AWS
```

## 📄 License

This Streamlit frontend is part of the L-MARS Legal Research System. See the main project license for details.