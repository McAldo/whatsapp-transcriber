# WhatsApp Transcriber - Technical Overview

## Document Purpose

This document provides a comprehensive technical overview of the WhatsApp Transcriber project for AI language models and developers who need to understand the architecture, implementation details, and design decisions to provide technical advice, debugging assistance, or feature implementations.

---

## Project Summary

**WhatsApp Transcriber** is a Python-based Streamlit web application that processes WhatsApp chat exports (ZIP files) and generates comprehensive timelines with automatic voice message transcriptions. The application emphasizes privacy, local processing, and optional AI enhancement with multiple LLM providers.

**Key Capabilities:**
- Parse WhatsApp chat export files (multiple date/time formats)
- Transcribe voice messages using local Whisper models or cloud APIs
- **Checkpoint & Resume system** - Automatic progress tracking with smart resume
- Enhance transcriptions using LLMs (Claude, OpenAI, Mistral AI, or local Ollama)
- Generate formatted timelines (Markdown, TXT, CSV, JSON)
- Organize and rename media files with timestamps
- Filter conversations by date range
- Mobile-friendly web interface with LAN access
- **UTF-8 encoding support** - Proper handling of all languages including Italian accents

---

## Technology Stack

### Core Framework
- **Streamlit 1.28.0+** - Web UI framework
- **Python 3.9+** - Programming language

### Transcription Engines
- **faster-whisper 0.10.0+** - Local speech-to-text (CTranslate2-based Whisper implementation)
  - Uses CUDA for GPU acceleration when available
  - Falls back to CPU if GPU unavailable
  - Models cached at `~/.cache/huggingface/hub/`

- **Voxtral Mini API** - Cloud-based transcription via Mistral API
  - Superior quality for Italian language
  - Cost: ~$0.001/minute

### LLM Providers (Optional)
- **Anthropic Claude API** (anthropic 0.7.0+)
- **OpenAI API** (openai 1.3.0+)
- **Mistral AI API** (mistralai 1.0.0+)
- **Ollama** (local, via HTTP requests)

### Supporting Libraries
- **pandas 2.0.0+** - Data manipulation for CSV export
- **python-dotenv 1.0.0+** - Environment variable management
- **tiktoken 0.5.0+** - Token counting for OpenAI models
- **requests 2.31.0+** - HTTP client for Ollama and Voxtral APIs

---

## Architecture Overview

### Directory Structure

```
whatsapp-transcriber/
├── app.py                          # Main Streamlit application
├── utils/
│   ├── __init__.py                # Module exports
│   ├── whatsapp_parser.py         # WhatsApp chat file parser
│   ├── audio_processor.py         # Faster-Whisper transcription
│   ├── voxtral_transcriber.py     # Voxtral Mini API transcription
│   ├── llm_corrector.py           # LLM correction with multiple providers
│   ├── file_organizer.py          # Media file organization and ZIP handling
│   ├── token_estimation.py        # Token counting and cost estimation
│   └── checkpoint_manager.py      # Checkpoint/resume functionality
├── chat_checkpoints/              # Checkpoint storage directory
├── check_checkpoint.py            # Checkpoint diagnostic tool
├── repair_checkpoint_encoding.py  # Encoding repair utility
├── requirements.txt               # Python dependencies
├── .env                          # Environment variables (API keys)
├── README.md                      # User documentation
├── DEVELOPER_README.md           # Developer guide
├── TECHNICAL_OVERVIEW.md         # This file
└── USER_GUIDE.md                 # Comprehensive user guide
```

### Data Flow

```
WhatsApp ZIP Export
      ↓
[CheckpointManager] Check for existing checkpoint (resume detection)
      ↓
[FileOrganizer] Extract ZIP → Parse chat file → Organize media
      ↓
[WhatsAppParser] Parse messages → Identify voice/text/media
      ↓
[Date Filter] Filter by date range (optional)
      ↓
[CheckpointManager] Create/load checkpoint → Track progress
      ↓
[AudioTranscriber/VoxtralTranscriber] Transcribe voice messages
      ↓   (Progress saved after each file)
[CheckpointManager] Update checkpoint → Save incrementally
      ↓
[LLMCorrector] Enhance transcriptions (optional, message-by-message or bulk)
      ↓
[CheckpointManager] Update checkpoint → Track LLM corrections
      ↓
[Timeline Generator] Format output (Markdown/TXT/CSV/JSON with UTF-8-BOM)
      ↓
[FileOrganizer] Create ZIP with media + timeline
      ↓
[CheckpointManager] Mark checkpoint as completed
      ↓
Download Results
```

---

## Core Components

### 1. Checkpoint Manager (`utils/checkpoint_manager.py`)

**Purpose:** Automatic progress tracking and resume functionality to prevent data loss on interruption.

**Key Class: `CheckpointManager`**

**Features:**
- Generates unique conversation IDs based on chat name and file count
- Creates checkpoints with full configuration and progress tracking
- Saves progress incrementally after each transcription
- Detects existing checkpoints for smart resume
- Fallback search by file count when exact ID doesn't match
- Tracks costs, errors, and processing log

**Checkpoint Structure:**
```python
{
    "conversation_id": str,          # Unique identifier
    "chat_name": str,                # Conversation name
    "timestamp": str,                # Last updated (ISO format)
    "total_files": int,              # Total audio files
    "processed_files": int,          # Successfully processed
    "failed_files": int,             # Failed transcriptions
    "transcriptions": {              # Results for each file
        "filename.opus": {
            "success": bool,
            "text": str,
            "corrected_text": str,   # If LLM used
            "language": str,
            "llm_corrected": bool,
            "duration": float,
            "cost": float
        }
    },
    "processing_log": [...],         # Event log
    "stats": {
        "total_transcription_cost": float,
        "total_llm_cost": float,
        "total_duration_minutes": float
    },
    "config": {                      # Original configuration
        "transcription_engine": str,
        "use_llm": bool,
        "llm_provider": str,
        "correction_mode": str
    },
    "completed": bool                # Marks full completion
}
```

**API Methods:**
```python
# Create new checkpoint
checkpoint = checkpoint_mgr.create_checkpoint(chat_data)

# Save checkpoint to disk
checkpoint_path = checkpoint_mgr.save_checkpoint(checkpoint)

# Find existing checkpoint
checkpoint_path = checkpoint_mgr.find_checkpoint_for_chat(conversation_id)

# Get pending files (not yet processed)
pending_files = checkpoint_mgr.get_pending_files(checkpoint, all_files)

# Get failed files (for retry)
failed_files = checkpoint_mgr.get_failed_files(checkpoint, all_files)

# Add transcription result
checkpoint_mgr.add_transcription_result(checkpoint, filename, result, file_info)

# Add log entry
checkpoint_mgr.add_log_entry(checkpoint, event_type, filename, metadata)
```

**Resume Logic:**
1. User uploads same ZIP file after interruption
2. System generates conversation ID from chat name + file count
3. Searches for existing checkpoint with matching ID
4. If exact match not found, fallback search by file count
5. Loads checkpoint and restores configuration
6. Identifies pending files (not in checkpoint['transcriptions'])
7. Processes only pending files
8. Updates existing checkpoint incrementally

---

### 2. WhatsApp Parser (`utils/whatsapp_parser.py`)

**Purpose:** Parse WhatsApp chat export text files into structured message objects.

**Key Class: `WhatsAppMessage`**
```python
class WhatsAppMessage:
    timestamp: datetime          # Message timestamp
    sender: str                  # Sender name
    content: str                 # Message text content
    message_type: str           # 'text', 'voice', 'image', 'video', 'document'
    media_file: str             # Path to media file (if applicable)
```

**Key Class: `WhatsAppParser`**
- **Supported timestamp formats:**
  - `[DD/MM/YYYY, HH:MM:SS]` (brackets)
  - `DD/MM/YYYY, HH:MM -` (dash separator)
  - `DD.MM.YY, HH:MM -` (German format)
  - `M/D/YY, H:MM AM/PM -` (US format)

- **Media detection patterns:**
  - Audio: `.opus`, `.mp3`, `.m4a`, `.aac`, `PTT-*-WA*.opus`
  - Images: `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`
  - Videos: `.mp4`, `.avi`, `.mov`, `.3gp`
  - Documents: `.pdf`, `.doc`, `.docx`, `.txt`, `.xlsx`

- **Multi-line message handling:** Continues building message content until next timestamp found

**Statistics tracked:**
- Total messages parsed
- Text vs media messages
- Parse errors

---

### 2. Audio Transcription

#### A. Faster-Whisper (`utils/audio_processor.py`)

**Key Class: `AudioTranscriber`**

**GPU Acceleration Logic:**
```python
# Attempts in order:
1. CUDA with float16 (fastest, modern GPUs)
2. CUDA with int8_float16 (mixed precision)
3. CUDA with int8 (all GPUs)
4. CPU fallback (if CUDA unavailable or fails)
```

**Error Handling:**
- Detects missing cuDNN libraries (`cudnn_ops64_9.dll` error)
- Sets `gpu_error` flag: `"cudnn_missing"`, `"cuda_unavailable"`, etc.
- Provides user-friendly error messages in UI

**Model Sizes:**
- tiny: ~75 MB, very fast, basic quality
- base: ~150 MB, fast, good quality (default)
- small: ~500 MB, medium speed, very good quality
- medium: ~1.5 GB, slow, excellent quality
- large: ~3 GB, very slow, best quality

**Transcription API:**
```python
result = transcriber.transcribe_file(audio_path, language=None)
# Returns:
{
    'success': bool,
    'text': str,
    'language': str,  # Detected language code
    'error': str      # If failed
}
```

**Performance Notes:**
- GPU: 5-10x faster than CPU
- Model caching: First use downloads model (~2-30 minutes depending on size)
- Subsequent uses: instant load from cache

#### B. Voxtral Mini API (`utils/voxtral_transcriber.py`)

**Key Class: `VoxtralTranscriber`**

**API Integration:**
- Endpoint: `https://api.mistral.ai/v1/audio/transcriptions`
- Model: `voxtral-mini-latest`
- Supports: `mp3`, `mp4`, `mpeg`, `mpga`, `m4a`, `wav`, `webm`, `opus`
- Max file size: 25 MB per file

**Cost Tracking:**
```python
# Automatically calculates duration and cost
transcriber.get_total_cost()  # Returns float (USD)
```

**Language Support:**
- Supports language hints via `language` parameter
- Superior quality for Italian language

---

### 3. LLM Correction (`utils/llm_corrector.py`)

**Purpose:** Post-process transcriptions to fix errors, improve punctuation, and enhance readability.

**Key Class: `LLMCorrector`**

**Initialization:**
```python
LLMCorrector(
    provider: str,              # 'claude', 'openai', 'mistral', 'ollama'
    api_key: str,              # API key (not needed for Ollama)
    ollama_url: str,           # Ollama server URL (default: localhost:11434)
    ollama_model: str,         # Ollama model name (default: llama3.2)
    correction_mode: str       # 'message' or 'bulk'
)
```

**Correction Modes:**

1. **Message-by-Message Mode** (`correction_mode='message'`)
   - Corrects each transcription individually
   - One API call per voice message
   - Safe for any conversation size
   - Uses smaller/cheaper models:
     - OpenAI: `gpt-4o-mini`
     - Mistral: `mistral-small-latest`
     - Claude: `claude-3-5-sonnet-20241022` (same for both)

2. **Bulk Mode** (`correction_mode='bulk'`)
   - Sends entire conversation as one transcript
   - Single API call for all messages
   - Requires large context window (100k+ tokens)
   - Uses larger models for better context understanding:
     - OpenAI: `gpt-4o`
     - Mistral: `mistral-large-latest`
     - Claude: `claude-3-5-sonnet-20241022` (same for both)

**Transcript Format (Bulk Mode):**
```
[TEXT - 2024-01-15 10:30:45] Alice: Hello, how are you?
[TRANSCRIPTION - 2024-01-15 10:31:20] Bob: I'm doing great thanks for asking
[TEXT - 2024-01-15 10:32:10] Alice: That's wonderful!
[TRANSCRIPTION - 2024-01-15 10:33:05] Bob: yeah it's been a good week
```

**Prompts:**
- `CORRECTION_PROMPT` - For individual messages
- `FULL_TRANSCRIPT_PROMPT` - For bulk corrections
- Both emphasize: minimal changes, preserve meaning, fix obvious errors only

**Provider-Specific Methods:**
```python
# Message-by-message
_correct_with_claude(text) -> str
_correct_with_openai(text) -> str
_correct_with_mistral(text) -> str
_correct_with_ollama(text) -> str

# Bulk mode
_correct_full_transcript_claude(transcript) -> str
_correct_full_transcript_openai(transcript) -> str
_correct_full_transcript_mistral(transcript) -> str
_correct_full_transcript_ollama(transcript) -> str
```

---

### 4. Token Estimation (`utils/token_estimation.py`)

**Purpose:** Estimate token counts and costs before processing to prevent errors and unexpected charges.

**Key Functions:**

```python
estimate_tokens(text: str, provider: str) -> int
```
- **OpenAI:** Uses `tiktoken` library with GPT-4 encoder (exact)
- **Claude:** ~3.5 characters per token (estimate)
- **Mistral:** ~4 characters per token (estimate)
- **Ollama:** ~4 characters per token (estimate)

```python
estimate_cost(token_count: int, provider: str, model: str, mode: str) -> float
```
- Calculates cost based on input/output token split (90% input, 10% output)
- Pricing per 1K tokens (as of 2025):
  - Claude Sonnet: $0.003 input, $0.015 output
  - GPT-4o: $0.0025 input, $0.01 output
  - GPT-4o-mini: $0.00015 input, $0.0006 output
  - Mistral Large: $0.008 input, $0.024 output
  - Mistral Small: $0.002 input, $0.006 output
  - Ollama: $0 (local)

```python
check_token_limit(token_count: int, provider: str, model: str) -> dict
```
- Context window limits:
  - Claude 3.5 Sonnet: 200,000 tokens
  - GPT-4o: 128,000 tokens
  - Mistral Large: 128,000 tokens
  - Mistral Small: 32,000 tokens
- Returns: `{'ok': bool, 'warning': bool, 'error': bool, 'message': str}`
- Warning at 80% of limit
- Error when exceeding limit

---

### 5. File Organization (`utils/file_organizer.py`)

**Key Class: `FileOrganizer`**

**Responsibilities:**
1. Extract WhatsApp export ZIP files
2. Locate chat text file (patterns: `*_chat.txt`, `*-chat.txt`, etc.)
3. Find and categorize media files
4. Rename media with timestamps: `YYYYMMDD_HHMMSS_originalname.ext`
5. Create organized output ZIP with media + timeline

**File Discovery:**
- Recursively searches extracted directory
- Identifies audio files: `.opus`, `.mp3`, `.m4a`, `.aac`
- Tracks statistics: total files, audio files, other media

**Media Organization:**
```python
organize_media(output_dir: str) -> dict
# Returns mapping: {original_filename: new_filename}
```

**Output ZIP Structure:**
```
WhatsApp_Chat_20240115-20240120_20250112_143022.zip
├── WhatsApp_Chat_20240115-20240120_20250112_143022.md
└── media/
    ├── 20240115_103045_PTT-20240115-WA0001.opus
    ├── 20240115_143522_IMG-20240115-WA0002.jpg
    ├── index.txt  # Maps new → original filenames
    └── ...
```

---

## Streamlit Application (`app.py`)

### Architecture Pattern

The Streamlit app follows a single-page application pattern with sections:

1. **Header** - Title, description, feature badges
2. **File Upload** - ZIP file upload with drag-and-drop
3. **Chat Preview** - Displays conversation info, participants, date range
4. **Date Range Filter** - Optional date filtering with quick presets
5. **Configuration** - Transcription, output, LLM settings
6. **Token Estimation** - Pre-processing estimates (bulk mode only)
7. **Process Button** - Triggers main processing pipeline
8. **Results** - Statistics and download buttons

### State Management

**Session State Variables:**
```python
st.session_state.uploaded_data = {
    'file_stats': dict,
    'chat_file': str,
    'date_info': dict,
    'messages': List[WhatsAppMessage],
    'audio_files': List[str]
}
st.session_state.current_file_name = str
st.session_state.date_filter_start = date  # For reactive updates
st.session_state.date_filter_end = date    # For reactive updates
```

### Configuration Schema

```python
config = {
    'transcription_engine': str,  # 'faster-whisper' or 'voxtral'
    'model_size': str,            # 'tiny', 'base', 'small', 'medium', 'large'
    'output_format': str,         # 'markdown', 'txt', 'csv', 'json'
    'output_filename': str,       # User-editable base filename
    'language': str,              # 'auto' or language code
    'use_llm': bool,              # Enable AI correction
    'llm_provider': str,          # 'claude', 'openai', 'mistral', 'ollama'
    'llm_api_key': str,           # API key for cloud providers
    'ollama_url': str,            # Ollama server URL
    'ollama_model': str,          # Ollama model name
    'mistral_api_key': str,       # Mistral API key (for Voxtral)
    'correction_mode': str,       # 'message' or 'bulk'
    'date_filter': dict           # Date filtering config
}
```

### Main Processing Pipeline (`process_chat()`)

```python
def process_chat(uploaded_file, config) -> dict:
    """
    Steps:
    1. Extract ZIP (10%)
    2. Parse chat file (20-30%)
    3. Apply date filtering (if enabled)
    4. Transcribe audio files (30-70%)
       - Progress updates per file
       - Language detection
       - Cost tracking (Voxtral)
    5. LLM correction (70-80%)
       - Message-by-message OR bulk mode
       - Cost estimation (bulk)
    6. Generate timeline (80-90%)
       - Format-specific generators
    7. Organize media (90-100%)
       - Rename files
       - Create output ZIP
    8. Return results with statistics
    """
```

**Progress Tracking:**
- `st.progress()` - Visual progress bar
- `st.empty()` - Dynamic status text updates
- Percentage-based updates throughout pipeline

**Error Handling:**
```python
results = {
    'success': bool,
    'stats': dict,
    'timeline_file': str,
    'media_zip': str,
    'processing_time': float,
    'errors': List[str]  # Accumulated errors
}
```

### Timeline Generators

Four format-specific generators:

1. **Markdown** (`generate_markdown_timeline()`)
   - Human-readable formatting
   - Timestamp headers
   - Bold sender names
   - Blockquotes for transcriptions
   - Emoji indicators (🎤, 📸, 🎥, 📄)

2. **Plain Text** (`generate_text_timeline()`)
   - Simple formatting
   - Clear section separators
   - [TRANSCRIPTION] tags

3. **CSV** (`generate_csv_timeline()`)
   - Spreadsheet-compatible
   - Columns: Timestamp, Sender, Type, Content, Media File, AI Enhanced, Language

4. **JSON** (`generate_json_timeline()`)
   - Programmatic access
   - Nested structure with metadata
   - Full transcription details

### Date Filtering

**UI Components:**
- 5 Quick preset buttons:
  - Last 1 Day
  - Last 7 Days
  - Last 30 Days
  - Last 90 Days
  - All Time
- Manual date range picker (start/end dates)
- Live message/audio count updates

**Implementation:**
```python
def filter_items_by_date(messages, audio_files, date_filter):
    """
    Filters both messages and audio files by date range.
    Matches audio files to their corresponding messages.
    Returns filtered lists + statistics.
    """
```

**Statistics Tracked:**
- Messages in range
- Audio files in range
- Skipped messages
- Skipped audio files
- Start/end dates

---

## UI/UX Patterns

### Reactive Updates (Session State Pattern)

Date filter quick buttons use session state for immediate UI updates:

```python
if st.button("Last 7 Days"):
    st.session_state.date_filter_start = today - timedelta(days=7)
    st.session_state.date_filter_end = max_date
    # Streamlit automatically re-renders, counts update
```

### Token Estimation Display (Bulk Mode)

Before processing, when bulk mode enabled:
```python
if config['correction_mode'] == 'bulk':
    # Build rough transcript
    # Estimate tokens
    # Check limits
    # Display:
    #   - Token count metric
    #   - Cost estimate
    #   - Warning/error if too large
```

### Provider Information Cards

Each LLM provider shows:
- Cost structure (Free/Paid)
- Privacy implications (Local/Cloud)
- Quality rating
- Speed rating
- Model selection (based on correction mode)

### Mobile Responsiveness

- Streamlit's responsive layout
- Column-based layouts collapse on mobile
- Touch-friendly buttons and controls
- No custom CSS required (Streamlit handles it)

---

## Security & Privacy

### API Key Management

**Recommended: Environment Variables**
```bash
# .env file
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
MISTRAL_API_KEY=...
```

**Fallback: UI Input**
- Password-type input field (masked)
- Not persisted across sessions
- Not logged or written to disk

### Data Privacy Levels

**Local Processing (No external calls):**
- Faster-Whisper transcription
- No LLM correction
- All data stays on device

**Partial Cloud (Transcription only):**
- Voxtral API: Audio files sent to Mistral
- No LLM correction
- Only audio transmitted

**Full Cloud (Transcription + Correction):**
- Voxtral + Claude/OpenAI/Mistral correction
- Audio and transcriptions sent to APIs
- Subject to provider privacy policies

**Hybrid (Local transcription, Local LLM):**
- Faster-Whisper + Ollama
- 100% local processing
- Best privacy option with AI enhancement

### Temporary File Cleanup

```python
# After processing:
organizer.cleanup()          # Removes extracted temp directory
os.unlink(zip_path)         # Removes uploaded ZIP temp file
# Output files kept in temp directory until download
```

---

## Performance Considerations

### GPU Acceleration (Faster-Whisper)

**Requirements:**
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- cuDNN libraries installed and in PATH

**Fallback Chain:**
1. Try CUDA float16 → fastest
2. Try CUDA int8_float16 → mixed precision
3. Try CUDA int8 → all GPUs
4. Fall back to CPU → slowest but works

**Common Issue: Missing cuDNN**
```
Error: "Could not locate cudnn_ops64_9.dll"
Solution: pip install nvidia-cudnn-cu12
```

### Model Caching

Whisper models cached at:
- **Windows:** `C:\Users\{user}\.cache\huggingface\hub\`
- **Mac/Linux:** `~/.cache/huggingface/hub/`

First use: Downloads model (2-30 minutes)
Subsequent uses: Instant load from cache

### LLM Correction Performance

**Message-by-Message:**
- 50 messages: ~50 API calls
- Processing time: ~2-5 minutes (depends on API latency)
- Network: Multiple round trips

**Bulk Mode:**
- 50 messages: 1 API call
- Processing time: ~10-30 seconds (depends on transcript size)
- Network: Single request/response
- **70-90% faster** for conversations with many messages

### Streamlit Performance

**Caching:**
```python
@st.cache_data
def extract_conversation_name(filename: str) -> str:
    # Cached to avoid re-computation on UI re-renders
```

**Avoiding Re-computation:**
- Parse chat file once, store in `st.session_state`
- Only re-filter when date range changes
- Use session state for reactive updates

---

## Error Handling Patterns

### Graceful Degradation

**GPU Error → CPU Fallback:**
```python
try:
    model = WhisperModel(size, device="cuda", compute_type="float16")
except Exception as e:
    logger.warning(f"CUDA float16 failed: {e}")
    # Try next compute type...
    # Eventually fall back to CPU
```

**LLM Error → Return Original:**
```python
try:
    corrected = corrector.correct_transcription(text)
except Exception as e:
    logger.error(f"Correction failed: {e}")
    corrected = text  # Return original if correction fails
```

### User-Friendly Error Messages

```python
if gpu_error == "cudnn_missing":
    st.warning(
        "⚠️ **GPU detected but cuDNN libraries missing - using CPU instead**\n\n"
        "Quick fix: Run `pip install nvidia-cudnn-cu12`\n\n"
        "See README for detailed instructions."
    )
```

### Validation Checks

**Pre-processing validation:**
```python
# Check API keys
if config['use_llm'] and config['llm_provider'] in ['claude', 'openai', 'mistral']:
    if not config['llm_api_key']:
        st.error(f"❌ API key required for {config['llm_provider'].title()}")
        return

# Check token limits (bulk mode)
if config['correction_mode'] == 'bulk':
    limit_check = check_token_limit(token_count, provider, model)
    if limit_check['error']:
        st.error(limit_check['message'])
        st.warning("⚠️ Switch to Message-by-Message mode to process this conversation.")
```

---

## Testing Considerations

### Unit Testing Targets

**WhatsAppParser:**
- Various timestamp format parsing
- Multi-line message handling
- Media attachment detection
- Edge cases: missing timestamps, malformed lines

**TokenEstimation:**
- Accuracy of estimates per provider
- Cost calculations with different models
- Limit checking edge cases

**LLMCorrector:**
- Provider initialization
- API error handling
- Response parsing

### Integration Testing

**End-to-end flow:**
1. Upload sample ZIP
2. Process with different configurations
3. Verify output format and content
4. Check media file organization

**Date filtering:**
1. Upload chat spanning multiple months
2. Apply various date ranges
3. Verify correct message/audio filtering

**LLM correction:**
1. Process with message-by-message mode
2. Process same chat with bulk mode
3. Compare results and costs

### Manual Testing Checklist

- [ ] GPU detection and fallback
- [ ] Model download (first run)
- [ ] Multiple output formats
- [ ] Date filtering accuracy
- [ ] LLM correction (all 4 providers)
- [ ] Token estimation accuracy
- [ ] Mobile responsiveness
- [ ] Error recovery (bad ZIP, missing chat file, etc.)
- [ ] API key validation
- [ ] Cost tracking accuracy

---

## Deployment Considerations

### Local Deployment

```bash
# Standard
streamlit run app.py

# With specific port
streamlit run app.py --server.port 8501

# Enable mobile access
streamlit run app.py --server.address 0.0.0.0
```

### Network Access

**Firewall configuration required for mobile access:**
- Port 8501 must be open on host machine
- Both devices on same network
- Access via `http://{host-ip}:8501`

### Resource Requirements

**Minimum:**
- Python 3.9+
- 4 GB RAM
- 2 GB disk (for small models)
- CPU-only (works but slow)

**Recommended:**
- Python 3.10+
- 8 GB RAM
- 5 GB disk (for medium models)
- NVIDIA GPU with 4+ GB VRAM
- CUDA 11.x or 12.x

**For large model (large-v2):**
- 16 GB RAM
- 10 GB disk
- GPU with 8+ GB VRAM

---

## Extension Points

### Adding New LLM Providers

1. Update `llm_corrector.py`:
```python
def _init_newprovider(self):
    # Initialize client

def _correct_with_newprovider(self, transcription: str) -> str:
    # Single message correction

def _correct_full_transcript_newprovider(self, transcript: str) -> str:
    # Bulk correction
```

2. Update `token_estimation.py`:
```python
# Add pricing
pricing['newprovider'] = {
    'model-name': {'input': 0.001, 'output': 0.003}
}

# Add context limits
limits['newprovider'] = {
    'model-name': 128000
}
```

3. Update `app.py`:
```python
# Add to provider dropdown
llm_provider = st.selectbox(
    "AI Provider",
    ['ollama', 'claude', 'openai', 'mistral', 'newprovider']
)

# Add API key handling
# Add model selection logic
```

### Adding New Output Formats

1. Create format generator:
```python
def generate_xml_timeline(messages, transcriptions, config, filter_stats):
    # Build XML structure
    return xml_string
```

2. Update `generate_timeline()`:
```python
elif output_format == 'xml':
    return generate_xml_timeline(messages, transcriptions, config, filter_stats)
```

3. Update UI:
```python
output_format = st.selectbox(
    "Timeline Format",
    ['markdown', 'txt', 'csv', 'json', 'xml']
)
```

### Adding New Transcription Engines

1. Create transcriber class in `utils/`:
```python
class NewTranscriber:
    def __init__(self, **kwargs):
        pass

    def initialize(self):
        pass

    def transcribe_file(self, audio_path: str, language: str = None) -> dict:
        return {
            'success': bool,
            'text': str,
            'language': str,
            'error': str
        }
```

2. Update `process_chat()` in `app.py`:
```python
if config['transcription_engine'] == 'newengine':
    transcriber = NewTranscriber(...)
    transcriber.initialize()
    # Process files...
```

3. Update UI configuration section

---

## Common Issues & Solutions

### Issue: "AttributeError: 'WhatsAppMessage' object has no attribute 'has_audio'"

**Cause:** Old code using wrong attribute names.

**Solution:** Use correct attributes:
- `message_type == 'voice'` instead of `has_audio`
- `media_file` instead of `audio_file`
- `content` instead of `text`

### Issue: GPU detected but using CPU

**Causes:**
1. Missing cuDNN libraries (most common)
2. Wrong CUDA version
3. CTranslate2 installed without CUDA support

**Solutions:**
1. Install cuDNN: `pip install nvidia-cudnn-cu12`
2. Add cuDNN to PATH (Windows)
3. Reinstall CTranslate2: `pip install ctranslate2 --extra-index-url https://pypi.nvidia.com`

### Issue: Token limit exceeded in bulk mode

**Cause:** Conversation too large for model's context window.

**Solution:**
- Switch to message-by-message mode
- Use a model with larger context (Claude 200k > GPT-4o 128k)
- Filter to smaller date range

### Issue: WhatsApp chat file not found in ZIP

**Cause:** Unexpected chat file naming or structure.

**Solution:** Update `FileOrganizer.get_chat_file()` patterns:
```python
CHAT_FILE_PATTERNS = [
    '*_chat.txt',
    '*-chat.txt',
    'WhatsApp Chat*.txt',
    # Add new pattern here
]
```

---

## API Documentation

### WhatsAppMessage

```python
class WhatsAppMessage:
    timestamp: datetime          # Message timestamp
    sender: str                  # Sender name
    content: str                 # Message text content
    message_type: str           # 'text', 'voice', 'image', 'video', 'document'
    media_file: Optional[str]   # Path to media file
```

### Transcription Result

```python
{
    'success': bool,                    # Whether transcription succeeded
    'text': str,                        # Transcribed text
    'corrected_text': str,             # LLM-corrected text (if applicable)
    'language': str,                    # Detected language code (e.g., 'en', 'it')
    'llm_corrected': bool,             # Whether LLM correction was applied
    'error': str                        # Error message (if failed)
}
```

### Process Results

```python
{
    'success': bool,
    'stats': {
        'total_messages': int,
        'text_messages': int,
        'voice_messages': int,
        'transcribed': int,
        'failed_transcriptions': int,
        'media_files': int,
        'transcription_engine': str,
        'transcription_cost': float,       # USD
        'language_mode': str,
        'detected_languages': str,
        'date_filtered': bool,
        'date_range': str,
        'skipped_by_filter': int,
        'llm_used': bool,
        'llm_provider': str,
        'llm_correction_mode': str,       # 'message' or 'bulk'
        'llm_correction_cost': float      # USD
    },
    'timeline_file': str,               # Path to generated timeline
    'media_zip': str,                   # Path to media ZIP
    'processing_time': float,           # Seconds
    'errors': List[str]                 # Accumulated errors
}
```

---

## Future Enhancement Ideas

### Performance Improvements
- [ ] Parallel audio transcription (process multiple files simultaneously)
- [ ] Streaming transcription for large files
- [ ] Incremental processing (resume from checkpoint)
- [ ] Database storage for large conversations

### Features
- [ ] Speaker diarization (identify different speakers in audio)
- [ ] Translation support (translate to different language)
- [ ] Sentiment analysis
- [ ] Keyword extraction and tagging
- [ ] Search functionality in processed timelines
- [ ] Web-based file browser (no download required)

### UI/UX
- [ ] Dark mode
- [ ] Progress persistence (survive page refresh)
- [ ] Batch processing (multiple chats)
- [ ] Comparison view (original vs corrected)
- [ ] Real-time streaming transcription preview

### Quality
- [ ] Transcription confidence scores
- [ ] Manual correction interface
- [ ] A/B testing different LLM providers
- [ ] Custom correction prompts

### Integration
- [ ] Direct WhatsApp API integration
- [ ] Telegram chat support
- [ ] Discord chat support
- [ ] Backup/restore functionality
- [ ] Cloud storage integration (Google Drive, Dropbox)

---

## Conclusion

This WhatsApp Transcriber application demonstrates a well-architected Python application with:
- Clean separation of concerns (modular utils)
- Graceful degradation and error handling
- User-friendly Streamlit interface
- Privacy-first design with optional cloud features
- Extensible architecture for new providers/formats
- Performance optimization (GPU, caching, bulk processing)

The codebase is suitable for educational purposes, personal use, and as a foundation for more advanced chat processing applications.

---

## Document Version

**Version:** 1.0
**Last Updated:** January 2025
**Maintained For:** AI Language Models and Technical Contributors
