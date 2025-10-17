# WhatsApp Transcriber - Developer Guide

> A production-ready Python application for processing WhatsApp exports with voice transcription and AI enhancement

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

WhatsApp Transcriber is a comprehensive solution for processing WhatsApp chat exports, automatically transcribing voice messages, and generating formatted timelines. Built with modern Python practices, it features a clean modular architecture, multiple AI provider integrations, and an intuitive Streamlit web interface.

### Key Highlights for Developers

- **🏗️ Modular Architecture** - Clean separation of concerns with reusable utility modules
- **🤖 Multi-Provider LLM Integration** - Abstract provider interface supporting Claude, OpenAI, Mistral AI, and Ollama
- **⚡ Performance Optimized** - GPU acceleration, model caching, bulk processing, and token estimation
- **🔄 Checkpoint & Resume System** - Automatic progress tracking with smart resume capabilities
- **🔒 Privacy-First Design** - Local processing by default, optional cloud features
- **📱 Mobile-Friendly** - Responsive web UI accessible from any device on LAN
- **🎯 Production-Ready** - Comprehensive error handling, graceful degradation, user-friendly feedback
- **🌍 Unicode Support** - Proper UTF-8 encoding with BOM for Windows compatibility

---

## Architecture

### Technology Stack

**Core:**
- Python 3.9+
- Streamlit 1.28+ (Web framework)
- faster-whisper 0.10+ (Speech-to-text via CTranslate2)

**AI/ML:**
- Anthropic Claude API
- OpenAI API
- Mistral AI API
- Ollama (local LLMs)
- tiktoken (token counting)

**Data Processing:**
- pandas (CSV export)
- python-dotenv (config management)
- requests (HTTP client)

### Project Structure

```
whatsapp-transcriber/
├── app.py                          # Main Streamlit application
├── utils/
│   ├── __init__.py                # Module exports
│   ├── whatsapp_parser.py         # Chat file parser
│   ├── audio_processor.py         # Whisper transcription
│   ├── voxtral_transcriber.py     # Mistral Voxtral API
│   ├── llm_corrector.py           # LLM correction (multi-provider)
│   ├── file_organizer.py          # ZIP/media handling
│   ├── token_estimation.py        # Token counting & cost estimation
│   └── checkpoint_manager.py      # Checkpoint/resume system
├── chat_checkpoints/              # Checkpoint storage directory
├── requirements.txt               # Dependencies
├── .env.example                   # Environment variable template
├── check_checkpoint.py            # Checkpoint diagnostic tool
├── repair_checkpoint_encoding.py  # Encoding repair utility
├── README.md                      # User documentation
├── DEVELOPER_README.md           # This file
├── TECHNICAL_OVERVIEW.md         # Detailed technical docs
└── USER_GUIDE.md                 # Comprehensive user guide
```

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) NVIDIA GPU with CUDA for faster transcription
- (Optional) API keys for AI enhancement

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/whatsapp-transcriber.git
cd whatsapp-transcriber
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment** (optional - for AI features)
```bash
# Copy example config
cp .env.example .env

# Edit .env and add your API keys
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
MISTRAL_API_KEY=...
```

5. **Run the application**
```bash
streamlit run app.py
```

Access at `http://localhost:8501`

---

## Development Guide

### Code Organization

#### `utils/whatsapp_parser.py`
**Purpose:** Parse WhatsApp export text files into structured data

**Key Classes:**
- `WhatsAppMessage` - Data class for messages
- `WhatsAppParser` - Main parser with regex patterns

**Features:**
- Supports multiple timestamp formats (US, EU, ISO)
- Multi-line message handling
- Media attachment detection (images, videos, audio, documents)
- Statistics tracking

**Usage:**
```python
from utils import WhatsAppParser

parser = WhatsAppParser()
messages = parser.parse_file("chat.txt")
stats = parser.get_statistics()
```

#### `utils/audio_processor.py`
**Purpose:** Local voice transcription using faster-whisper

**Key Features:**
- Automatic GPU detection with fallback to CPU
- Multiple compute type attempts (float16 → int8 → CPU)
- Model caching for fast subsequent runs
- Detailed error reporting (e.g., missing cuDNN)

**Usage:**
```python
from utils import AudioTranscriber

transcriber = AudioTranscriber(model_size='base')
transcriber.load_model()

result = transcriber.transcribe_file("audio.opus", language="en")
# Returns: {'success': bool, 'text': str, 'language': str, 'error': str}

transcriber.unload_model()  # Free memory
```

#### `utils/llm_corrector.py`
**Purpose:** Abstract LLM correction interface supporting multiple providers

**Design Pattern:** Strategy pattern for provider selection

**Key Features:**
- Unified interface for 4 providers (Claude, OpenAI, Mistral, Ollama)
- Two correction modes: message-by-message or bulk
- Automatic model selection based on mode
- Provider-specific error handling

**Usage:**
```python
from utils import LLMCorrector

corrector = LLMCorrector(
    provider='claude',
    api_key='sk-ant-...',
    correction_mode='bulk'
)
corrector.initialize()

# Single message
corrected = corrector.correct_transcription("original text")

# Full transcript (bulk mode)
corrected_transcript = corrector.correct_full_transcript(full_text)
```

**Adding New Providers:**

1. Add initialization method:
```python
def _init_newprovider(self):
    from newprovider import Client
    self.client = Client(api_key=self.api_key)
```

2. Add correction methods:
```python
def _correct_with_newprovider(self, transcription: str) -> str:
    response = self.client.complete(prompt=...)
    return response.text

def _correct_full_transcript_newprovider(self, transcript: str) -> str:
    response = self.client.complete(prompt=...)
    return response.text
```

3. Update `initialize()` and `correct_transcription()` methods

#### `utils/checkpoint_manager.py`
**Purpose:** Automatic progress tracking and resume functionality

**Key Features:**
- Creates checkpoints for each conversation
- Saves progress after every transcription
- Tracks costs, processed files, and errors
- Enables smart resume on interruption
- Fallback search by file count for robustness

**Usage:**
```python
from utils import CheckpointManager

checkpoint_mgr = CheckpointManager(checkpoint_dir='chat_checkpoints')

# Create new checkpoint
chat_data = {
    'conversation_id': 'unique_id',
    'chat_name': 'Chat Name',
    'total_files': 100,
    'config': {...}
}
checkpoint = checkpoint_mgr.create_checkpoint(chat_data)

# Save progress
checkpoint_mgr.add_transcription_result(checkpoint, filename, result, file_info)
checkpoint_mgr.save_checkpoint(checkpoint, checkpoint_path)

# Find existing checkpoint
checkpoint_path = checkpoint_mgr.find_checkpoint_for_chat(conversation_id)

# Get pending files
pending_files = checkpoint_mgr.get_pending_files(checkpoint, all_files)
```

#### `utils/token_estimation.py`
**Purpose:** Pre-processing cost estimation and limit checking

**Key Functions:**
```python
estimate_tokens(text: str, provider: str) -> int
# Uses tiktoken for OpenAI, character estimates for others

estimate_cost(token_count: int, provider: str, model: str, mode: str) -> float
# Calculates cost based on provider pricing (input/output split)

check_token_limit(token_count: int, provider: str, model: str) -> dict
# Returns: {'ok': bool, 'warning': bool, 'error': bool, 'message': str}

format_token_count(token_count: int) -> str
# Pretty formatting with thousands separator
```

**Pricing Data:**
Update pricing dictionary when provider prices change:
```python
pricing = {
    'claude': {
        'claude-3-5-sonnet-20241022': {'input': 0.003, 'output': 0.015}
    },
    # ... add new models/providers here
}
```

#### `utils/file_organizer.py`
**Purpose:** ZIP extraction, media organization, and output packaging

**Features:**
- Automatic chat file detection (multiple naming patterns)
- Media file discovery and categorization
- Timestamp-based renaming
- Index generation for file mapping

---

### Extending the Application

#### Adding New Output Formats

1. **Create format generator in `app.py`:**
```python
def generate_html_timeline(messages, transcriptions, config, filter_stats):
    """Generate HTML timeline."""
    html = ['<!DOCTYPE html><html><body>']

    # Build HTML from messages...

    html.append('</body></html>')
    return '\n'.join(html)
```

2. **Update `generate_timeline()` router:**
```python
def generate_timeline(messages, transcriptions, output_format, config, filter_stats):
    if output_format == 'markdown':
        return generate_markdown_timeline(...)
    elif output_format == 'html':
        return generate_html_timeline(...)
    # ... other formats
```

3. **Add to UI dropdown:**
```python
output_format = st.selectbox(
    "Timeline Format",
    ['markdown', 'txt', 'csv', 'json', 'html']
)
```

#### Adding New Transcription Engines

1. **Create engine module in `utils/`:**
```python
# utils/new_transcriber.py
class NewTranscriber:
    def __init__(self, **config):
        self.config = config

    def initialize(self):
        # Initialize engine
        pass

    def transcribe_file(self, audio_path: str, language: str = None) -> dict:
        # Transcribe audio
        return {
            'success': True,
            'text': 'transcribed text',
            'language': 'en',
            'error': None
        }
```

2. **Update `process_chat()` in `app.py`:**
```python
if config['transcription_engine'] == 'new_engine':
    from utils.new_transcriber import NewTranscriber

    transcriber = NewTranscriber(**config)
    transcriber.initialize()

    for audio_file in audio_files:
        result = transcriber.transcribe_file(audio_file)
        transcriptions[os.path.basename(audio_file)] = result
```

3. **Update UI configuration section**

---

## Key Features Deep Dive

### 1. GPU Acceleration

**Implementation:** Progressive fallback chain

```python
def load_model(self):
    compute_types = ['float16', 'int8_float16', 'int8']

    for compute_type in compute_types:
        try:
            self.model = WhisperModel(
                self.model_size,
                device="cuda",
                compute_type=compute_type
            )
            self.device = "cuda"
            return
        except Exception as e:
            logger.warning(f"CUDA {compute_type} failed: {e}")
            continue

    # Final fallback
    self.model = WhisperModel(self.model_size, device="cpu")
    self.device = "cpu"
```

**Error Detection:**
- Identifies missing cuDNN libraries
- Provides actionable fix instructions in UI
- Gracefully degrades to CPU

### 2. Smart Correction Modes

**Message-by-Message:**
- Sequential API calls
- Safe for any conversation size
- More expensive for large conversations
- Uses smaller models (gpt-4o-mini, mistral-small)

**Bulk Mode:**
- Single API call for entire conversation
- 70-90% faster
- Requires large context window (100k+ tokens)
- Uses larger models (gpt-4o, mistral-large)
- Pre-processing token estimation with limit checking

**Implementation:**
```python
if correction_mode == "bulk":
    # Build full transcript
    transcript = build_full_transcript(messages, transcriptions)

    # Estimate and validate
    token_count = estimate_tokens(transcript, provider)
    limit_check = check_token_limit(token_count, provider, model)

    if limit_check['error']:
        # Abort or fallback to message-by-message
        raise ValueError("Transcript too large")

    # Single API call
    corrected = corrector.correct_full_transcript(transcript)

    # Parse back to individual messages
    parse_corrected_transcript(corrected, transcriptions)
else:
    # Message-by-message
    for filename, trans_result in transcriptions.items():
        corrected = corrector.correct_transcription(trans_result['text'])
        trans_result['corrected_text'] = corrected
```

### 3. Date Range Filtering

**Implementation:** Session state for reactive updates

```python
# Initialize session state
if 'date_filter_start' not in st.session_state:
    st.session_state.date_filter_start = min_date

# Quick preset button
if st.button("Last 7 Days"):
    st.session_state.date_filter_start = today - timedelta(days=7)
    st.session_state.date_filter_end = max_date
    # Streamlit auto-reruns, counts update immediately

# Date inputs read from session state
start_date = st.date_input(
    "Start Date",
    value=st.session_state.date_filter_start,
    min_value=min_date,
    max_value=max_date
)

# Update session state when manually changed
st.session_state.date_filter_start = start_date

# Counts recalculate on every render
counts = count_items_in_date_range(messages, audio_files, start_date, end_date)
st.metric("Messages", counts['messages'])
```

### 4. Cost Estimation

**Pre-Processing Display (Bulk Mode):**
```python
if config['correction_mode'] == 'bulk':
    # Build rough transcript
    rough_transcript = build_estimate_transcript(messages)

    # Estimate tokens
    token_count = estimate_tokens(rough_transcript, provider)

    # Calculate cost
    cost = estimate_cost(token_count, provider, model, mode='bulk')

    # Display before user clicks "Process"
    st.metric("Estimated Tokens", f"{token_count:,}")
    st.metric("Estimated Cost", f"${cost:.4f}")
```

**Post-Processing Tracking:**
```python
# In process_chat()
llm_correction_cost = 0.0

if correction_mode == 'bulk':
    # After correction completes
    actual_token_count = estimate_tokens(full_transcript, provider)
    llm_correction_cost = estimate_cost(actual_token_count, provider, model)

# Include in results
results['stats']['llm_correction_cost'] = llm_correction_cost
```

---

## API Reference

### Core Data Structures

#### WhatsAppMessage
```python
class WhatsAppMessage:
    timestamp: datetime          # Message timestamp
    sender: str                  # Sender name
    content: str                 # Message text
    message_type: str           # 'text', 'voice', 'image', 'video', 'document'
    media_file: Optional[str]   # Path to media file
```

#### Transcription Result
```python
{
    'success': bool,                # Transcription succeeded
    'text': str,                    # Original transcription
    'corrected_text': str,         # LLM-corrected (if applicable)
    'language': str,                # Language code ('en', 'it', etc.)
    'llm_corrected': bool,         # LLM correction applied
    'error': Optional[str]          # Error message if failed
}
```

#### Configuration
```python
{
    'transcription_engine': str,    # 'faster-whisper' | 'voxtral'
    'model_size': str,              # 'tiny' | 'base' | 'small' | 'medium' | 'large'
    'output_format': str,           # 'markdown' | 'txt' | 'csv' | 'json'
    'output_filename': str,         # Base filename
    'language': str,                # 'auto' | language code
    'use_llm': bool,                # Enable AI correction
    'llm_provider': str,            # 'claude' | 'openai' | 'mistral' | 'ollama'
    'llm_api_key': str,            # API key
    'correction_mode': str,         # 'message' | 'bulk'
    'date_filter': dict            # Date filtering config
}
```

---

## Recent Improvements (January 2025)

### Checkpoint & Resume System
- Automatic progress saving after each transcription
- Smart resume on interruption (crash, restart, close app)
- Settings restoration from checkpoint
- Cost tracking across sessions
- Fallback checkpoint detection by file count
- Diagnostic tools: `check_checkpoint.py`, `repair_checkpoint_encoding.py`

### Encoding Fixes
- UTF-8-BOM encoding for Windows compatibility
- Proper handling of Italian accents (è, à, ò, ù, ì)
- Encoding fix function in Voxtral transcriber
- Checkpoint encoding validation and repair tool

### Bug Fixes
- **Progress bar overflow** - Fixed double counting causing >100% progress
- **Processing time calculation** - Fixed variable collision causing TypeError
- **Checkpoint detection** - Improved reliability with fallback search
- **Config restoration** - Settings now automatically restored on resume
- **LLM correction issues** - Fixed corrupted input causing identical outputs

---

## Testing

### Manual Testing Checklist

- [ ] Upload various WhatsApp export formats (Android, iOS)
- [ ] Test all output formats (MD, TXT, CSV, JSON)
- [ ] Verify date filtering accuracy
- [ ] Test LLM correction with all providers
- [ ] Validate token estimation accuracy
- [ ] Check GPU detection and fallback
- [ ] Test mobile responsiveness
- [ ] Verify error handling (bad files, missing keys, etc.)
- [ ] Check cost tracking accuracy
- [ ] Test checkpoint/resume functionality
- [ ] Verify encoding for Italian/special characters
- [ ] Test settings restoration on resume

### Unit Testing (Recommended)

```python
# tests/test_parser.py
def test_parse_us_format():
    parser = WhatsAppParser()
    # Test timestamp parsing...

def test_multiline_messages():
    # Test multi-line message handling...

# tests/test_token_estimation.py
def test_openai_token_count():
    # Test tiktoken accuracy...

def test_cost_calculation():
    # Test pricing calculations...
```

---

## Performance Optimization

### Transcription Performance

**GPU vs CPU (base model):**
- GPU (CUDA): ~5-10x faster
- GPU (CUDA) with cuDNN: ~10-15x faster
- CPU: Baseline (1x)

**Model Selection Trade-offs:**
| Model | Speed | Quality | Memory | Use Case |
|-------|-------|---------|--------|----------|
| tiny | 10x | Basic | 1 GB | Testing, clear audio |
| base | 5x | Good | 2 GB | Production (recommended) |
| small | 2x | Very Good | 4 GB | Important chats |
| medium | 1x | Excellent | 8 GB | Critical accuracy |
| large | 0.5x | Best | 12 GB | Professional use |

### LLM Correction Performance

**Message-by-Message (50 messages):**
- API calls: 50
- Time: ~2-5 minutes
- Cost: ~$0.10 (varies by provider)

**Bulk Mode (50 messages):**
- API calls: 1
- Time: ~10-30 seconds
- Cost: ~$0.03 (varies by provider)
- **Improvement: 70-90% faster, 70% cheaper**

### Streamlit Optimization

```python
# Cache expensive operations
@st.cache_data
def load_and_parse_chat(file_path: str):
    parser = WhatsAppParser()
    return parser.parse_file(file_path)

# Use session state to avoid recomputation
if 'parsed_messages' not in st.session_state:
    st.session_state.parsed_messages = parse_chat(file)
```

---

## Deployment

### Local Deployment

**Standard:**
```bash
streamlit run app.py
```

**With Custom Port:**
```bash
streamlit run app.py --server.port 8080
```

**Enable Mobile Access:**
```bash
streamlit run app.py --server.address 0.0.0.0
```

Access from mobile: `http://{computer-ip}:8501`

### Docker Deployment (Optional)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for GPU support (optional)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

**Build and run:**
```bash
docker build -t whatsapp-transcriber .
docker run -p 8501:8501 -v ~/.cache:/root/.cache whatsapp-transcriber
```

### Cloud Deployment Considerations

**Streamlit Community Cloud:**
- Free hosting
- No GPU support (CPU-only transcription)
- API keys via Streamlit secrets
- Public or password-protected

**AWS/GCP/Azure:**
- Full control
- GPU instances available
- Higher costs
- Better performance

---

## Troubleshooting

### Common Issues

**1. GPU not detected despite having NVIDIA GPU**

Check:
```bash
# Verify CUDA
nvidia-smi

# Install cuDNN
pip install nvidia-cudnn-cu12

# Reinstall CTranslate2 with GPU support
pip install ctranslate2 --force-reinstall --extra-index-url https://pypi.nvidia.com
```

**2. Model download fails or times out**

Solution:
```bash
# Pre-download model
python -c "from faster_whisper import WhisperModel; WhisperModel('base')"
```

**3. API key not recognized**

Check:
- `.env` file in project root
- Correct variable names (ANTHROPIC_API_KEY, OPENAI_API_KEY, MISTRAL_API_KEY)
- No quotes around values
- Restart application after changing .env

**4. "Token limit exceeded" in bulk mode**

Solutions:
- Switch to message-by-message mode
- Use model with larger context (Claude 200k > GPT-4o 128k)
- Filter conversation to smaller date range

---

## Contributing

### Areas for Contribution

**High Priority:**
- [ ] Unit tests for core modules
- [ ] Integration tests
- [ ] Additional timestamp format support
- [ ] Performance benchmarks
- [ ] Documentation improvements

**Feature Requests:**
- [ ] Speaker diarization
- [ ] Translation support
- [ ] Sentiment analysis
- [ ] Batch processing
- [ ] Web-based file browser

**Code Quality:**
- [ ] Type hints throughout
- [ ] Docstring coverage
- [ ] Code formatting (black, isort)
- [ ] Linting (pylint, flake8)

### Development Workflow

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes with descriptive commits
4. Add tests if applicable
5. Update documentation
6. Submit pull request with clear description

---

## License

MIT License - See LICENSE file for details

---

## Resources

**Documentation:**
- [Streamlit Docs](https://docs.streamlit.io)
- [faster-whisper](https://github.com/guillaumekln/faster-whisper)
- [Anthropic Claude API](https://docs.anthropic.com)
- [OpenAI API](https://platform.openai.com/docs)
- [Mistral AI API](https://docs.mistral.ai)

**Related Projects:**
- [Whisper (Original)](https://github.com/openai/whisper)
- [CTranslate2](https://github.com/OpenNMT/CTranslate2)
- [Ollama](https://ollama.ai)

---

## Support

**Issues:** [GitHub Issues](https://github.com/yourusername/whatsapp-transcriber/issues)

**Discussions:** [GitHub Discussions](https://github.com/yourusername/whatsapp-transcriber/discussions)

---

**Built with ❤️ for the developer community**
