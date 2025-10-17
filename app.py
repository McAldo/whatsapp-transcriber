"""
WhatsApp Transcriber - Main Streamlit Application
A tool to process WhatsApp chat exports and create comprehensive timelines with transcriptions.
"""

import streamlit as st
import os
import sys
import tempfile
import logging
import json
import csv
from datetime import datetime, timedelta
from pathlib import Path
import socket
from dotenv import load_dotenv

# Import utilities
from utils import (
    WhatsAppParser, AudioTranscriber, LLMCorrector, FileOrganizer,
    get_model_info, get_provider_info, create_output_zip, is_model_cached, get_gpu_status,
    VoxtralTranscriber, get_audio_duration, calculate_total_duration
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="WhatsApp Transcriber",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for mobile responsiveness
st.markdown("""
<style>
    /* Mobile-friendly styling */
    @media (max-width: 768px) {
        .stButton>button {
            width: 100%;
            margin: 5px 0;
        }

        .uploadedFile {
            font-size: 14px;
        }
    }

    /* Full-width buttons */
    .stButton>button {
        min-height: 50px;
        font-size: 16px;
        font-weight: 600;
    }

    /* Larger touch targets */
    .stCheckbox, .stRadio {
        padding: 10px 0;
    }

    /* Better spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Stats card styling */
    .stats-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }

    /* Progress styling */
    .status-text {
        font-size: 14px;
        color: #666;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)


def get_local_ip():
    """Get the local IP address of the machine."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "Unable to determine"


def initialize_session_state():
    """Initialize session state variables."""
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'timeline_file' not in st.session_state:
        st.session_state.timeline_file = None
    if 'media_zip' not in st.session_state:
        st.session_state.media_zip = None
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'current_file_name' not in st.session_state:
        st.session_state.current_file_name = None
    if 'checkpoint_found' not in st.session_state:
        st.session_state.checkpoint_found = False
    if 'checkpoint_path' not in st.session_state:
        st.session_state.checkpoint_path = None
    if 'resume_checkpoint' not in st.session_state:
        st.session_state.resume_checkpoint = None
    if 'processing_mode' not in st.session_state:
        st.session_state.processing_mode = None


def analyze_date_range(messages, audio_files):
    """
    Analyze date range from messages and audio files.

    Args:
        messages: List of WhatsAppMessage objects
        audio_files: List of audio file paths

    Returns:
        Dictionary with date range info
    """
    from utils.audio_processor import AudioTranscriber

    dates = []

    # Get dates from text messages
    for msg in messages:
        if msg.timestamp:
            dates.append(msg.timestamp.date())

    # Get dates from audio filenames
    transcriber = AudioTranscriber()
    for audio_file in audio_files:
        filename = os.path.basename(audio_file)
        timestamp = transcriber.extract_timestamp_from_filename(filename)
        if timestamp:
            dates.append(timestamp.date())
        else:
            # Try file modification time as fallback
            file_time = transcriber.get_file_creation_time(audio_file)
            if file_time:
                dates.append(file_time.date())

    if not dates:
        return {
            'has_dates': False,
            'min_date': None,
            'max_date': None
        }

    return {
        'has_dates': True,
        'min_date': min(dates),
        'max_date': max(dates),
        'total_dates': len(dates)
    }


def count_items_in_date_range(messages, audio_files, start_date, end_date):
    """
    Count items within a date range.

    Args:
        messages: List of WhatsAppMessage objects
        audio_files: List of audio file paths
        start_date: Start date (datetime.date object)
        end_date: End date (datetime.date object)

    Returns:
        Dictionary with counts
    """
    from utils.audio_processor import AudioTranscriber

    text_count = 0
    audio_count = 0
    media_count = 0

    # Count text messages
    for msg in messages:
        if msg.timestamp and start_date <= msg.timestamp.date() <= end_date:
            if msg.message_type == 'text':
                text_count += 1
            elif msg.message_type != 'voice':
                media_count += 1

    # Count audio files
    transcriber = AudioTranscriber()
    for audio_file in audio_files:
        filename = os.path.basename(audio_file)
        timestamp = transcriber.extract_timestamp_from_filename(filename)
        if not timestamp:
            file_time = transcriber.get_file_creation_time(audio_file)
            timestamp = file_time if file_time else None

        if timestamp and start_date <= timestamp.date() <= end_date:
            audio_count += 1

    return {
        'text_messages': text_count,
        'audio_files': audio_count,
        'media_files': media_count
    }


def render_sidebar():
    """Render sidebar with network access instructions."""
    with st.sidebar:
        st.title("📱 Mobile Access")

        local_ip = get_local_ip()
        port = 8501  # Default Streamlit port

        st.markdown(f"""
        ### Access from your phone:

        1. Connect phone to same WiFi network
        2. Open browser on phone
        3. Go to: `http://{local_ip}:{port}`

        ---

        ### Find your IP address:

        **Windows:**
        ```
        ipconfig
        ```
        Look for "IPv4 Address"

        **Mac/Linux:**
        ```
        ifconfig
        ```
        Look for "inet" address

        ---

        **Note:** Your computer's firewall must allow connections on port {port}
        """)


def extract_conversation_name(filename):
    """
    Extract conversation name from WhatsApp export filename.

    Args:
        filename: Original ZIP filename (e.g., "WhatsApp Chat with Name Surname.zip")

    Returns:
        Extracted name or "WhatsApp_Chat" as fallback
    """
    import re

    # Remove .zip extension
    name = filename.replace('.zip', '')

    # Common patterns:
    # "WhatsApp Chat with Name Surname"
    # "WhatsApp Chat - Name Surname"
    # "Chat de WhatsApp con Name Surname"

    patterns = [
        r'WhatsApp\s+Chat\s+with\s+(.+)',  # English
        r'WhatsApp\s+Chat\s+-\s+(.+)',     # Alternative English
        r'Chat\s+de\s+WhatsApp\s+con\s+(.+)',  # Spanish
        r'Conversazione\s+WhatsApp\s+con\s+(.+)',  # Italian
        r'Chat\s+WhatsApp\s+avec\s+(.+)',  # French
        r'WhatsApp\s+Chat\s+mit\s+(.+)',   # German
    ]

    for pattern in patterns:
        match = re.search(pattern, name, re.IGNORECASE)
        if match:
            # Clean up the name
            extracted = match.group(1).strip()
            # Replace invalid filename characters
            extracted = re.sub(r'[<>:"/\\|?*]', '_', extracted)
            return extracted

    # Fallback: use cleaned version of filename
    cleaned = re.sub(r'[<>:"/\\|?*]', '_', name)
    return cleaned if cleaned else "WhatsApp_Chat"


def generate_default_filename(uploaded_filename, date_filter=None):
    """
    Generate default transcript filename.

    Args:
        uploaded_filename: Original uploaded file name
        date_filter: Date filter dictionary (optional)

    Returns:
        Default filename without extension
    """
    # Extract conversation name
    conv_name = extract_conversation_name(uploaded_filename)

    # Add date range if filtered
    if date_filter and date_filter.get('enabled') and date_filter.get('start_date') and date_filter.get('end_date'):
        start = date_filter['start_date'].strftime('%Y%m%d')
        end = date_filter['end_date'].strftime('%Y%m%d')
        date_range = f"_{start}-{end}"
    else:
        date_range = ""

    # Add transcription date and time
    now = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Combine all parts
    filename = f"{conv_name}{date_range}_{now}"

    return filename


def render_upload_section():
    """Render file upload section."""
    st.header("📦 Upload WhatsApp Export")

    uploaded_file = st.file_uploader(
        "Choose a WhatsApp export ZIP file",
        type=['zip'],
        help="Export your WhatsApp chat (with media) and upload the ZIP file here"
    )

    if uploaded_file:
        file_size = uploaded_file.size / (1024 * 1024)  # Convert to MB
        st.success(f"✓ File uploaded: {uploaded_file.name} ({file_size:.2f} MB)")

        if file_size > 500:
            st.warning("⚠️ Large file detected. Processing may take some time.")

    return uploaded_file


def render_date_range_filter(date_info, messages, audio_files):
    """
    Render date range filter section.

    Args:
        date_info: Dictionary with date range information
        messages: List of WhatsAppMessage objects
        audio_files: List of audio file paths

    Returns:
        Dictionary with filter settings
    """
    if not date_info['has_dates']:
        return {
            'enabled': False,
            'start_date': None,
            'end_date': None
        }

    with st.expander("📅 Date Range Filter (Optional)", expanded=False):
        # Show conversation date range
        st.info(f"**Conversation spans:** {date_info['min_date'].strftime('%Y-%m-%d')} to {date_info['max_date'].strftime('%Y-%m-%d')}")

        # Process entire conversation checkbox
        process_all = st.checkbox("Process entire conversation", value=True)

        # Initialize session state for date filtering if needed
        if 'date_filter_start' not in st.session_state:
            st.session_state.date_filter_start = date_info['min_date']
        if 'date_filter_end' not in st.session_state:
            st.session_state.date_filter_end = date_info['max_date']

        start_date = date_info['min_date']
        end_date = date_info['max_date']

        if not process_all:
            # Quick preset buttons (show these first for better UX)
            st.subheader("Quick Filters")
            col_a, col_b, col_c, col_d, col_e = st.columns(5)

            today = datetime.now().date()

            with col_a:
                if st.button("Last 1 Day", key="filter_1day"):
                    st.session_state.date_filter_start = max(today - timedelta(days=1), date_info['min_date'])
                    st.session_state.date_filter_end = date_info['max_date']

            with col_b:
                if st.button("Last 7 Days", key="filter_7days"):
                    st.session_state.date_filter_start = max(today - timedelta(days=7), date_info['min_date'])
                    st.session_state.date_filter_end = date_info['max_date']

            with col_c:
                if st.button("Last 30 Days", key="filter_30days"):
                    st.session_state.date_filter_start = max(today - timedelta(days=30), date_info['min_date'])
                    st.session_state.date_filter_end = date_info['max_date']

            with col_d:
                if st.button("Last 3 Months", key="filter_3months"):
                    st.session_state.date_filter_start = max(today - timedelta(days=90), date_info['min_date'])
                    st.session_state.date_filter_end = date_info['max_date']

            with col_e:
                if st.button("Last Year", key="filter_1year"):
                    st.session_state.date_filter_start = max(today - timedelta(days=365), date_info['min_date'])
                    st.session_state.date_filter_end = date_info['max_date']

            st.divider()
            st.subheader("Custom Date Range")

            col1, col2 = st.columns(2)

            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=st.session_state.date_filter_start,
                    min_value=date_info['min_date'],
                    max_value=date_info['max_date'],
                    key="start_date_input"
                )
                # Update session state when manually changed
                st.session_state.date_filter_start = start_date

            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=st.session_state.date_filter_end,
                    min_value=date_info['min_date'],
                    max_value=date_info['max_date'],
                    key="end_date_input"
                )
                # Update session state when manually changed
                st.session_state.date_filter_end = end_date

            # Validate dates
            if start_date > end_date:
                st.error("❌ Start date must be before or equal to end date!")
                return {
                    'enabled': True,
                    'start_date': None,
                    'end_date': None,
                    'valid': False
                }

        # Show item count
        counts = count_items_in_date_range(messages, audio_files, start_date, end_date)

        st.success(
            f"**Will process:** {counts['text_messages']} text messages, "
            f"{counts['audio_files']} audio files, {counts['media_files']} media files"
        )

        if not process_all:
            total_msgs = len([m for m in messages if m.message_type == 'text'])
            total_audio = len(audio_files)
            skipped_text = total_msgs - counts['text_messages']
            skipped_audio = total_audio - counts['audio_files']

            if skipped_text > 0 or skipped_audio > 0:
                st.warning(
                    f"⚠️ **Excluded:** {skipped_text} text messages, {skipped_audio} audio files "
                    f"(outside selected date range)"
                )

    return {
        'enabled': not process_all,
        'start_date': start_date,
        'end_date': end_date,
        'valid': True
    }


def render_configuration_section():
    """Render configuration section."""
    with st.expander("⚙️ Configuration", expanded=True):
        # Transcription engine selection
        st.subheader("🎤 Transcription Engine")

        transcription_engine = st.selectbox(
            "Engine",
            ['voxtral', 'faster-whisper'],
            format_func=lambda x: {
                'faster-whisper': 'Faster-Whisper (Local)',
                'voxtral': 'Voxtral Mini (Mistral API)'
            }[x],
            index=0,  # Default to Voxtral (recommended)
            help="Choose transcription engine. Voxtral recommended for best quality."
        )

        # Show info message for Voxtral
        if transcription_engine == 'voxtral':
            st.info("💡 Voxtral offers better quality, especially for Italian. Cost: ~$0.001/minute (~$0.06/hour)")

            # Show comparison table
            with st.expander("ℹ️ Which engine should I choose?"):
                st.markdown("""
                | Feature | Faster-Whisper | Voxtral Mini API |
                |---------|----------------|------------------|
                | Quality | Good | Better |
                | Italian | Good | Excellent |
                | Speed | Fast | Fast |
                | Cost | Free | ~$0.001/min |
                | Privacy | 100% local | Sends to API |
                | Internet | Not required | Required |
                | Setup | Included | Need API key |
                """)

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            # Whisper model selection (only for faster-whisper)
            if transcription_engine == 'faster-whisper':
                st.subheader("Transcription Model")
                model_sizes = ['tiny', 'base', 'small', 'medium', 'large']

                model_size = st.selectbox(
                    "Whisper Model",
                    model_sizes,
                    index=1,  # Default to 'base'
                    help="Select transcription model size"
                )
            else:
                # For Voxtral, model is fixed
                st.subheader("Transcription Model")
                st.info("🔧 Model: voxtral-mini-latest (automatic)")
                model_size = None  # Not used for Voxtral

            # Show model info (only for faster-whisper)
            if transcription_engine == 'faster-whisper':
                model_info = get_model_info(model_size)
                st.caption(f"**Speed:** {model_info['speed']} | **Quality:** {model_info['quality']}")
                st.caption(f"**Size:** {model_info['size']}")
                st.caption(f"ℹ️ {model_info['description']}")

                # Check if model is cached
                if is_model_cached(model_size):
                    st.success(f"✓ {model_size.title()} model already downloaded")
                else:
                    st.info(f"⚠️ First use will download ~{model_info['size']} (may take a few minutes)")

                # Show GPU status
                st.markdown("**Hardware Acceleration:**")
                gpu_status = get_gpu_status()
                if gpu_status['available']:
                    st.success(gpu_status['message'])
                    if gpu_status['device_names']:
                        for device in gpu_status['device_names']:
                            st.caption(f"  • {device}")
                    if gpu_status.get('note'):
                        st.caption(f"ℹ️ {gpu_status['note']}")
                else:
                    st.warning(gpu_status['message'])
                    if gpu_status['backend'] == 'CTranslate2 (CPU)':
                        with st.expander("💡 Enable GPU for faster transcription (5-10x speedup!)"):
                            st.markdown("""
                            **Quick Setup Guide:**

                            **Step 1:** Verify CUDA is installed:
                            ```bash
                            nvidia-smi
                            ```
                            (Should show CUDA version 11.x or 12.x)

                            **Step 2:** Install cuDNN via pip:
                            ```bash
                            # For CUDA 12.x (most common):
                            pip install nvidia-cudnn-cu12

                            # For CUDA 11.x:
                            pip install nvidia-cudnn-cu11
                            ```

                            **Step 3:** Add cuDNN to PATH (Windows only):
                            ```bash
                            # Find cuDNN location:
                            python -c "import nvidia.cudnn; import os; print(os.path.dirname(nvidia.cudnn.__file__))"

                            # Add the shown path + "\\bin" to your System PATH
                            # (Search "Environment Variables" → Edit Path → Add New)
                            ```

                            **Step 4:** Install GPU-enabled ctranslate2:
                            ```bash
                            pip install ctranslate2 --force-reinstall --extra-index-url https://pypi.nvidia.com
                            ```

                            **Step 5:** Restart the app

                            ---

                            **Requirements:** NVIDIA GPU with CUDA support

                            **Still not working?** See detailed troubleshooting in README.md
                            """)
            else:
                # For Voxtral, show API-based info
                st.caption("ℹ️ Voxtral uses Mistral's cloud API - no local GPU required")

        with col2:
            # Output format selection
            st.subheader("Output Format")
            output_format = st.selectbox(
                "Timeline Format",
                ['markdown', 'txt', 'csv', 'json'],
                index=0,
                help="Choose format for timeline output"
            )

        # Output filename editor
        st.divider()
        st.subheader("📝 Output Filename")

        # Get uploaded file name from session state for default
        uploaded_file_name = st.session_state.get('current_file_name', None)

        if uploaded_file_name:
            # Generate default filename
            default_name = extract_conversation_name(uploaded_file_name)

            st.caption(f"ℹ️ Extracted from: {uploaded_file_name}")
        else:
            default_name = "WhatsApp_Chat"
            st.caption("ℹ️ Upload a file to see extracted name")

        output_filename = st.text_input(
            "Base filename (without extension)",
            value=default_name,
            help="This will be used as the base name. Date range and timestamp will be added automatically."
        )

        # Show preview of final filename
        if output_filename:
            preview_with_datetime = f"{output_filename}_[date_range]_YYYYMMDD_HHMMSS"
            st.caption(f"**Preview:** `{preview_with_datetime}.{output_format}`")
            st.caption("*Note: [date_range] will be added only if date filtering is enabled*")

        # Language selection
        st.divider()
        st.subheader("🌍 Transcription Language")

        language_options = {
            'auto': 'Auto-detect (Recommended)',
            'en': 'English',
            'it': 'Italian',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'pt': 'Portuguese',
            'nl': 'Dutch',
            'ru': 'Russian',
            'zh': 'Chinese',
            'ja': 'Japanese'
        }

        language_choice = st.selectbox(
            "Language",
            options=list(language_options.keys()),
            format_func=lambda x: language_options[x],
            index=0,  # Default to auto-detect
            help="Select the language of voice messages. Auto-detect works for most cases."
        )

        if language_choice == 'auto':
            st.caption("ℹ️ Whisper will automatically detect the language in each voice message")
        else:
            st.caption(f"ℹ️ All voice messages will be transcribed as {language_options[language_choice]}")

        st.divider()

        # LLM Correction toggle
        st.subheader("✨ AI Transcription Enhancement (Optional)")
        use_llm = st.checkbox(
            "Enable AI correction",
            value=False,
            help="Use AI to improve transcription quality, fix errors, and add punctuation"
        )

        llm_provider = None
        llm_api_key = None
        ollama_url = None
        ollama_model = None
        correction_mode = "message"  # Default

        if use_llm:
            # Correction mode selector
            st.subheader("Correction Mode")
            correction_mode = st.radio(
                "Choose correction method:",
                ["message", "bulk"],
                format_func=lambda x: {
                    "message": "Message-by-Message (Slower, any model)",
                    "bulk": "Full Transcript (Faster, needs large context)"
                }[x],
                help="Message-by-Message: Corrects each voice message individually. Full Transcript: Sends entire conversation at once (faster but requires large context window).",
                horizontal=True
            )

            if correction_mode == "message":
                st.caption("ℹ️ Corrects each transcription individually. Slower but works with any model.")
            else:
                st.caption("ℹ️ Sends entire transcript at once. Faster and cheaper, but requires large context window (100k+ tokens).")

            st.divider()

            col3, col4 = st.columns(2)

            with col3:
                llm_provider = st.selectbox(
                    "AI Provider",
                    ['openai', 'claude', 'mistral', 'ollama'],
                    format_func=lambda x: {
                        'ollama': '🔒 Ollama (Local - FREE & PRIVATE)',
                        'claude': 'Claude API (Anthropic)',
                        'openai': 'OpenAI API (ChatGPT)',
                        'mistral': 'Mistral AI'
                    }[x],
                    index=0,  # Default to OpenAI (recommended)
                    help="Choose your AI provider. OpenAI recommended for best quality."
                )

                # Show provider info
                provider_info = get_provider_info(llm_provider)
                st.caption(f"**Cost:** {provider_info['cost']}")
                st.caption(f"**Privacy:** {provider_info['privacy']}")

            with col4:
                if llm_provider == 'ollama':
                    # Ollama configuration
                    ollama_url = st.text_input(
                        "Ollama URL",
                        value="http://localhost:11434",
                        help="URL of your Ollama server"
                    )

                    ollama_model = st.selectbox(
                        "Model",
                        ['llama3.2', 'llama3.1', 'mistral', 'gemma2'],
                        help="Select Ollama model (must be pulled first)"
                    )

                    # Test connection button
                    if st.button("🔌 Test Ollama Connection"):
                        try:
                            import requests
                            response = requests.get(f"{ollama_url}/api/tags", timeout=5)
                            response.raise_for_status()
                            data = response.json()
                            models = [m['name'] for m in data.get('models', [])]

                            if any(ollama_model in m for m in models):
                                st.success(f"✓ Connected! Model '{ollama_model}' is available.")
                            else:
                                st.error(f"❌ Model '{ollama_model}' not found. Available models: {', '.join(models)}")
                                st.info(f"Run: `ollama pull {ollama_model}`")
                        except Exception as e:
                            st.error(f"❌ Connection failed: {e}")
                            st.info("Make sure Ollama is running. Install from: https://ollama.ai")

                else:
                    # API key handling for Claude/OpenAI/Mistral
                    if llm_provider == 'claude':
                        env_var = 'ANTHROPIC_API_KEY'
                    elif llm_provider == 'openai':
                        env_var = 'OPENAI_API_KEY'
                    elif llm_provider == 'mistral':
                        env_var = 'MISTRAL_API_KEY'
                    else:
                        env_var = 'API_KEY'

                    env_key = os.getenv(env_var)

                    if env_key:
                        st.success(f"✓ API key loaded from environment ({env_var})")
                        llm_api_key = env_key
                    else:
                        st.info(f"💡 Set {env_var} in .env file or enter below")
                        llm_api_key = st.text_input(
                            "API Key",
                            type="password",
                            help=f"Enter your {llm_provider.title()} API key"
                        )

                    # Show which model will be used
                    if llm_provider == 'claude':
                        model_name = "claude-3-5-sonnet-20241022"
                        st.caption(f"📦 Model: {model_name}")
                    elif llm_provider == 'openai':
                        model_name = "gpt-4o" if correction_mode == "bulk" else "gpt-4o-mini"
                        st.caption(f"📦 Model: {model_name}")
                    elif llm_provider == 'mistral':
                        model_name = "mistral-large-latest" if correction_mode == "bulk" else "mistral-small-latest"
                        st.caption(f"📦 Model: {model_name}")

        # Mistral API key handling for Voxtral
        st.divider()
        mistral_api_key = None

        if transcription_engine == 'voxtral':
            st.subheader("🔑 Mistral API Key")

            # Check for environment variable
            env_key = os.getenv('MISTRAL_API_KEY')

            if env_key:
                st.success(f"✓ API key loaded from environment (MISTRAL_API_KEY)")
                mistral_api_key = env_key
            else:
                st.info(f"💡 Set MISTRAL_API_KEY in .env file or enter below")
                mistral_api_key = st.text_input(
                    "Mistral API Key",
                    type="password",
                    help="Enter your Mistral API key. Get one at https://console.mistral.ai/"
                )

                if not mistral_api_key:
                    st.warning("⚠️ Mistral API key required for Voxtral transcription")
                    st.markdown("""
                    **To get your API key:**
                    1. Sign up at [console.mistral.ai](https://console.mistral.ai/)
                    2. Navigate to API Keys section
                    3. Create a new key
                    4. Copy and paste it above
                    """)

            # Test connection button
            if mistral_api_key:
                if st.button("🔌 Test Mistral API Connection"):
                    with st.spinner("Testing connection..."):
                        try:
                            test_transcriber = VoxtralTranscriber(api_key=mistral_api_key)
                            result = test_transcriber.test_connection()

                            if result['success']:
                                st.success(f"✓ {result['message']}")
                            else:
                                st.error(f"❌ {result['message']}")
                                if 'error' in result:
                                    st.caption(f"Details: {result['error']}")
                        except Exception as e:
                            st.error(f"❌ Connection test failed: {e}")

        # Processing Settings
        st.divider()
        with st.expander("⚙️ Processing Settings (Advanced)", expanded=False):
            st.markdown("**Resilience & Recovery Settings**")
            st.caption("Configure timeout handling and checkpoint behavior")

            col_a, col_b = st.columns(2)

            with col_a:
                file_timeout = st.number_input(
                    "File Timeout (seconds)",
                    min_value=60,
                    max_value=600,
                    value=180,
                    step=30,
                    help="Maximum time to wait per file before skipping. Recommended: 180s (3 minutes)"
                )

                auto_resume = st.checkbox(
                    "Auto-resume interrupted processing",
                    value=False,
                    help="Automatically continue from last checkpoint without prompting"
                )

            with col_b:
                save_uncorrected = st.checkbox(
                    "Save both corrected & uncorrected timelines",
                    value=True,
                    help="Generate both versions for comparison (recommended)"
                )

                st.caption("💡 **Checkpoints save progress** automatically after each file")
                st.caption("⚠️ You can safely interrupt processing (Ctrl+C) and resume later")

    return {
        'transcription_engine': transcription_engine,
        'model_size': model_size,
        'output_format': output_format,
        'output_filename': output_filename,
        'language': language_choice,
        'use_llm': use_llm,
        'llm_provider': llm_provider,
        'llm_api_key': llm_api_key,
        'ollama_url': ollama_url,
        'ollama_model': ollama_model,
        'mistral_api_key': mistral_api_key,
        'correction_mode': correction_mode,
        'file_timeout': file_timeout,
        'auto_resume': auto_resume,
        'save_uncorrected': save_uncorrected
    }


def filter_items_by_date(messages, audio_files, date_filter):
    """
    Filter messages and audio files by date range.

    Args:
        messages: List of WhatsAppMessage objects
        audio_files: List of audio file paths
        date_filter: Date filter configuration

    Returns:
        Tuple of (filtered_messages, filtered_audio_files, stats)
    """
    from utils.audio_processor import AudioTranscriber

    if not date_filter or not date_filter.get('enabled'):
        return messages, audio_files, {'filtered': False}

    start_date = date_filter['start_date']
    end_date = date_filter['end_date']

    # Filter messages
    filtered_messages = []
    skipped_messages = 0

    for msg in messages:
        if msg.timestamp and start_date <= msg.timestamp.date() <= end_date:
            filtered_messages.append(msg)
        else:
            skipped_messages += 1

    # Filter audio files
    filtered_audio = []
    skipped_audio = 0
    transcriber = AudioTranscriber()

    for audio_file in audio_files:
        filename = os.path.basename(audio_file)
        timestamp = transcriber.extract_timestamp_from_filename(filename)
        if not timestamp:
            file_time = transcriber.get_file_creation_time(audio_file)
            timestamp = file_time if file_time else None

        if timestamp and start_date <= timestamp.date() <= end_date:
            filtered_audio.append(audio_file)
        else:
            skipped_audio += 1

    return filtered_messages, filtered_audio, {
        'filtered': True,
        'start_date': start_date,
        'end_date': end_date,
        'skipped_messages': skipped_messages,
        'skipped_audio': skipped_audio
    }


def process_chat(uploaded_file, config, checkpoint=None, retry_mode=False):
    """
    Main processing function with checkpoint support.

    Args:
        uploaded_file: Uploaded ZIP file
        config: Configuration dictionary
        checkpoint: Optional existing checkpoint to resume from
        retry_mode: If True, only retry failed files from checkpoint

    Returns:
        Results dictionary
    """
    results = {
        'success': False,
        'stats': {},
        'timeline_file': None,
        'media_zip': None,
        'processing_time': 0,
        'errors': []
    }

    start_time = datetime.now()

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Initialize checkpoint manager
    from utils import CheckpointManager, get_file_info, process_with_timeout, log_processing_metrics
    checkpoint_mgr = CheckpointManager()
    checkpoint_path = None
    is_resuming = checkpoint is not None

    # Stats tracking
    stats_cols = st.columns(4)
    completed_metric = stats_cols[0].empty()
    failed_metric = stats_cols[1].empty()
    cost_metric = stats_cols[2].empty()
    time_metric = stats_cols[3].empty()

    try:
        # Step 1: Extract files (10%)
        status_text.markdown("**📦 Extracting files...**")
        progress_bar.progress(10)

        # Save uploaded file to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            zip_path = tmp_file.name

        organizer = FileOrganizer()
        temp_dir = organizer.extract_zip(zip_path)

        file_stats = organizer.get_statistics()
        status_text.markdown(f"**📦 Extracted {file_stats['total_files']} files**")
        progress_bar.progress(20)

        # Step 2: Parse chat file (30%)
        status_text.markdown("**📝 Parsing WhatsApp chat...**")

        chat_file = organizer.get_chat_file()
        if not chat_file:
            raise ValueError("Could not find WhatsApp chat text file in ZIP")

        parser = WhatsAppParser()
        messages = parser.parse_file(chat_file)
        parse_stats = parser.get_statistics()
        audio_files = organizer.get_audio_files()

        status_text.markdown(f"**📝 Parsed {len(messages)} messages**")

        # Apply date filtering if enabled
        date_filter = config.get('date_filter')
        filter_stats = {'filtered': False}

        if date_filter and date_filter.get('enabled'):
            status_text.markdown("**📅 Applying date range filter...**")
            messages, audio_files, filter_stats = filter_items_by_date(
                messages, audio_files, date_filter
            )
            if filter_stats['filtered']:
                status_text.markdown(
                    f"**📅 Filtered to date range: {filter_stats['start_date']} to {filter_stats['end_date']}**\n\n"
                    f"Skipped: {filter_stats['skipped_messages']} messages, {filter_stats['skipped_audio']} audio files"
                )

        status_text.markdown(f"**📝 Processing {len(messages)} messages, {len(audio_files)} audio files**")
        progress_bar.progress(30)

        # Initialize or resume checkpoint
        if is_resuming:
            status_text.markdown("**📋 Resuming from checkpoint...**")
            # Checkpoint already loaded, just get path
            checkpoint_path = checkpoint_mgr.save_checkpoint(checkpoint)  # Get path

            # Restore critical config settings from checkpoint
            if 'config' in checkpoint and checkpoint['config']:
                saved_config = checkpoint['config']
                logger.info(f"Restoring config from checkpoint: {saved_config}")

                # Update current config with saved settings (for key processing parameters)
                if 'transcription_engine' in saved_config:
                    config['transcription_engine'] = saved_config['transcription_engine']
                if 'use_llm' in saved_config:
                    config['use_llm'] = saved_config['use_llm']
                if 'llm_provider' in saved_config:
                    config['llm_provider'] = saved_config['llm_provider']
                if 'correction_mode' in saved_config:
                    config['correction_mode'] = saved_config['correction_mode']

                status_text.info(
                    f"📋 Restored settings from checkpoint:\n\n"
                    f"- Engine: {saved_config.get('transcription_engine', 'N/A')}\n\n"
                    f"- LLM: {saved_config.get('use_llm', False)}\n\n"
                    f"- Mode: {saved_config.get('correction_mode', 'N/A')}"
                )
        else:
            # Create new checkpoint
            conv_name = extract_conversation_name(uploaded_file.name) if hasattr(uploaded_file, 'name') else "Chat"
            # Date range for display (stored in checkpoint metadata)
            if filter_stats.get('filtered'):
                date_range_display = f"{filter_stats['start_date']} to {filter_stats['end_date']}"
            else:
                date_range_display = "all"

            chat_data = {
                # Always use "all" for conversation ID to ensure consistency
                'conversation_id': checkpoint_mgr.generate_conversation_id(conv_name, "all", len(audio_files)),
                'chat_name': conv_name,
                'total_files': len(audio_files),
                'date_range': date_range_display,
                'config': {
                    'transcription_engine': config['transcription_engine'],
                    'use_llm': config['use_llm'],
                    'llm_provider': config.get('llm_provider'),
                    'correction_mode': config.get('correction_mode', 'message')
                }
            }

            checkpoint = checkpoint_mgr.create_checkpoint(chat_data)
            checkpoint_path = checkpoint_mgr.save_checkpoint(checkpoint)
            status_text.markdown(f"**✓ Checkpoint created:** {checkpoint['conversation_id'][:8]}...")

        # Get list of files to process
        if retry_mode:
            # Retry mode: only process failed files
            pending_audio_files = checkpoint_mgr.get_failed_files(checkpoint, audio_files)
            status_text.markdown(f"**🔁 Retrying: {len(pending_audio_files)} failed files**")
            # Remove failed files from checkpoint so they can be retried
            for file_path in pending_audio_files:
                filename = os.path.basename(file_path)
                if filename in checkpoint['transcriptions']:
                    # Keep record but mark for retry
                    checkpoint['transcriptions'][filename]['retry'] = True
                    checkpoint['failed_files'] -= 1
                    checkpoint['processed_files'] -= 1
        elif is_resuming:
            # Resume mode: process pending files
            pending_audio_files = checkpoint_mgr.get_pending_files(checkpoint, audio_files)
            status_text.markdown(f"**▶️ Resuming: {len(pending_audio_files)} files remaining**")
        else:
            # Fresh mode: process all files
            pending_audio_files = audio_files

        # Update initial metrics
        completed_metric.metric("Completed", checkpoint['processed_files'])
        failed_metric.metric("Failed", checkpoint['failed_files'])
        cost_metric.metric("Cost", f"${checkpoint['stats']['total_transcription_cost']:.4f}")
        time_metric.metric("Time", "0s")

        # Capture initial processed count (to avoid double counting in progress calculation)
        initial_processed_count = checkpoint['processed_files']

        # Step 3: Transcribe audio files (30-70%)
        # Load existing transcriptions from checkpoint (if resuming)
        transcriptions = checkpoint.get('transcriptions', {}).copy()
        logger.info(f"Loaded {len(transcriptions)} existing transcriptions from checkpoint")
        total_cost = 0.0

        if audio_files:
            # Choose transcription engine
            if config['transcription_engine'] == 'voxtral':
                # Voxtral Mini API transcription
                status_text.markdown(f"**🎤 Initializing Voxtral Mini API...**")

                # Calculate estimated cost upfront (skip if too many files to avoid long wait)
                if len(audio_files) <= 50:
                    status_text.markdown("**⏱️ Calculating audio duration for cost estimate...**")
                    total_duration = calculate_total_duration(audio_files)
                    estimated_cost = (total_duration / 60.0) * 0.001
                    status_text.info(
                        f"**💰 Estimated cost:** ${estimated_cost:.4f} USD\n\n"
                        f"Total audio: {total_duration / 60.0:.1f} minutes (~{len(audio_files)} files)"
                    )
                else:
                    # For large batches, show approximate estimate based on average
                    status_text.info(
                        f"**💰 Estimated cost:** ~${len(audio_files) * 0.0001:.4f} USD (approximate)\n\n"
                        f"~{len(audio_files)} audio files (assuming ~6 seconds average per file)\n\n"
                        f"*Note: Exact cost calculation skipped for large batches to avoid delays*"
                    )

                try:
                    transcriber = VoxtralTranscriber(api_key=config['mistral_api_key'])
                    transcriber.initialize()

                    status_text.markdown(f"**✓ Voxtral Mini ready! Transcribing {len(audio_files)} audio files...**")

                    # Prepare language parameter
                    lang_param = None if config['language'] == 'auto' else config['language']

                    for idx, audio_file in enumerate(pending_audio_files, 1):
                        filename = os.path.basename(audio_file)

                        # Calculate progress based on total files (not just pending)
                        total_processed = initial_processed_count + idx
                        progress = 30 + int((total_processed / checkpoint['total_files']) * 40)
                        # Cap progress at 70 to avoid exceeding 100
                        progress = min(progress, 70)
                        progress_bar.progress(progress)

                        # Log start event
                        checkpoint_mgr.add_log_entry(checkpoint, "transcription_start", filename)

                        # Collect file info
                        file_info = get_file_info(audio_file)

                        status_text.markdown(
                            f"**🎤 Transcribing with Voxtral Mini: {total_processed}/{checkpoint['total_files']}**\n\n"
                            f"File: {filename}\n\n"
                            f"Size: {file_info['size_bytes'] / 1024:.1f} KB | Progress: {progress}%"
                        )

                        # Transcribe with timeout
                        import time
                        file_start_time = time.time()

                        result = process_with_timeout(
                            transcriber.transcribe_file,
                            (audio_file,),
                            timeout_seconds=config.get('file_timeout', 180),
                            kwargs={'language': lang_param}
                        )

                        elapsed = time.time() - file_start_time

                        # Add to checkpoint
                        checkpoint_mgr.add_transcription_result(checkpoint, filename, result, file_info)

                        # Log completion or failure
                        if result.get('success'):
                            checkpoint_mgr.add_log_entry(checkpoint, "transcription_complete", filename, {
                                "duration": elapsed,
                                "text_length": len(result.get('text', ''))
                            })
                        else:
                            checkpoint_mgr.add_log_entry(checkpoint, "transcription_failed", filename, {
                                "duration": elapsed,
                                "error": result.get('error', 'Unknown error')
                            })

                        # Save checkpoint after each file
                        checkpoint_mgr.save_checkpoint(checkpoint, checkpoint_path)

                        # Update metrics
                        completed_metric.metric("Completed", f"{checkpoint['processed_files']}/{checkpoint['total_files']}")
                        failed_metric.metric("Failed", checkpoint['failed_files'])
                        cost_metric.metric("Cost", f"${checkpoint['stats']['total_transcription_cost']:.4f}")
                        time_metric.metric("Time", f"{elapsed:.1f}s")

                        # Log metrics
                        log_processing_metrics(filename, file_info, result, elapsed)

                        # Store in transcriptions dict (for timeline generation)
                        transcriptions[filename] = result

                        # Update status with detected language if available
                        if result.get('success') and result.get('language'):
                            detected_lang = result['language']
                            status_text.markdown(
                                f"**🎤 Transcribing with Voxtral Mini: {total_processed}/{checkpoint['total_files']}**\n\n"
                                f"File: {filename} | Language: {detected_lang.upper()}\n\n"
                                f"Progress: {progress}%"
                            )

                    # Get total cost
                    total_cost = transcriber.get_total_cost()
                    status_text.success(f"**✓ Transcription complete! Total cost: ${total_cost:.4f} USD**")

                except Exception as e:
                    logger.error(f"Voxtral transcription failed: {e}")
                    results['errors'].append(f"Voxtral transcription failed: {e}")
                    status_text.error(f"❌ Voxtral transcription failed: {e}")
                    return results

            else:
                # Faster-Whisper (local) transcription
                status_text.markdown(f"**🎤 Preparing transcription model ({config['model_size']})...**")

                # Note: First-time model download can take several minutes
                transcriber = AudioTranscriber(model_size=config['model_size'])

                status_text.markdown(
                    f"**⏳ Loading {config['model_size']} model...**\n\n"
                    f"*Note: First-time use will download the model (~{get_model_info(config['model_size'])['size']}). "
                    f"This may take a few minutes depending on your internet speed.*"
                )

                transcriber.load_model()

                # Check if GPU failed and show helpful message
                if transcriber.gpu_error == "cudnn_missing":
                    status_text.warning(
                        "⚠️ **GPU detected but cuDNN libraries missing - using CPU instead**\n\n"
                        "Your GPU is available but cuDNN (CUDA Deep Neural Network library) is not properly installed. "
                        "Transcription will be slower on CPU.\n\n"
                        "**Quick fix:** Run `pip install nvidia-cudnn-cu12` (or cu11 for CUDA 11.x)\n\n"
                        "Then add cuDNN to PATH and restart the app. See README for detailed instructions."
                    )
                elif transcriber.device == "cpu" and transcriber.gpu_error:
                    status_text.warning(
                        f"⚠️ **GPU acceleration unavailable - using CPU**\n\n"
                        "Transcription will be slower. See README for GPU setup instructions."
                    )
                else:
                    status_text.markdown(f"**✓ Model loaded on {transcriber.device.upper()}! Transcribing {len(audio_files)} audio files...**")

                # Prepare language parameter
                lang_param = None if config['language'] == 'auto' else config['language']

                for idx, audio_file in enumerate(pending_audio_files, 1):
                    filename = os.path.basename(audio_file)

                    # Calculate progress based on total files (not just pending)
                    total_processed = initial_processed_count + idx
                    progress = 30 + int((total_processed / checkpoint['total_files']) * 40)
                    # Cap progress at 70 to avoid exceeding 100
                    progress = min(progress, 70)
                    progress_bar.progress(progress)

                    # Log start event
                    checkpoint_mgr.add_log_entry(checkpoint, "transcription_start", filename)

                    # Collect file info
                    file_info = get_file_info(audio_file)

                    status_text.markdown(
                        f"**🎤 Transcribing {total_processed}/{checkpoint['total_files']}: {filename}**\n\n"
                        f"Size: {file_info['size_bytes'] / 1024:.1f} KB | Progress: {progress}%"
                    )

                    # Transcribe with timeout
                    import time
                    file_start_time = time.time()

                    result = process_with_timeout(
                        transcriber.transcribe_file,
                        (audio_file,),
                        timeout_seconds=config.get('file_timeout', 180),
                        kwargs={'language': lang_param}
                    )

                    elapsed = time.time() - file_start_time

                    # Add to checkpoint
                    checkpoint_mgr.add_transcription_result(checkpoint, filename, result, file_info)

                    # Log completion or failure
                    if result.get('success'):
                        checkpoint_mgr.add_log_entry(checkpoint, "transcription_complete", filename, {
                            "duration": elapsed,
                            "text_length": len(result.get('text', ''))
                        })
                    else:
                        checkpoint_mgr.add_log_entry(checkpoint, "transcription_failed", filename, {
                            "duration": elapsed,
                            "error": result.get('error', 'Unknown error')
                        })

                    # Save checkpoint after each file
                    checkpoint_mgr.save_checkpoint(checkpoint, checkpoint_path)

                    # Update metrics
                    completed_metric.metric("Completed", f"{checkpoint['processed_files']}/{checkpoint['total_files']}")
                    failed_metric.metric("Failed", checkpoint['failed_files'])
                    cost_metric.metric("Cost", f"${checkpoint['stats']['total_transcription_cost']:.4f}")
                    time_metric.metric("Time", f"{elapsed:.1f}s")

                    # Log metrics
                    log_processing_metrics(filename, file_info, result, elapsed)

                    # Store in transcriptions dict (for timeline generation)
                    transcriptions[filename] = result

                    # Update status with detected language if auto-detect
                    if config['language'] == 'auto' and result.get('success') and result.get('language'):
                        detected_lang = result['language']
                        status_text.markdown(
                            f"**🎤 Transcribing {total_processed}/{checkpoint['total_files']}: {filename}**\n\n"
                            f"Progress: {progress}% | Detected: {detected_lang.upper()}"
                        )

                transcriber.unload_model()

        progress_bar.progress(70)

        # Step 4: LLM correction (70-80%)
        llm_correction_cost = 0.0
        correction_mode = config.get('correction_mode', 'message')

        if config['use_llm'] and transcriptions:
            status_text.markdown(f"**✨ Applying AI corrections ({correction_mode} mode)...**")

            try:
                corrector = LLMCorrector(
                    provider=config['llm_provider'],
                    api_key=config['llm_api_key'],
                    ollama_url=config['ollama_url'],
                    ollama_model=config['ollama_model'],
                    correction_mode=correction_mode
                )
                corrector.initialize()

                if correction_mode == "bulk":
                    # Bulk mode: Send entire transcript at once
                    status_text.markdown("**✨ Building full transcript for bulk correction...**")

                    # Build full transcript
                    transcript_lines = []
                    filename_to_message = {}  # Track which line corresponds to which file

                    for msg in messages:
                        timestamp_str = msg.timestamp.strftime("%Y-%m-%d %H:%M:%S") if msg.timestamp else "Unknown"

                        if msg.message_type == 'voice' and msg.media_file:
                            filename = os.path.basename(msg.media_file)
                            trans_result = transcriptions.get(filename)

                            if trans_result and trans_result.get('success') and trans_result.get('text'):
                                content = trans_result['text']
                                line = f"[TRANSCRIPTION - {timestamp_str}] {msg.sender}: {content}"
                                transcript_lines.append(line)
                                filename_to_message[len(transcript_lines) - 1] = filename
                        else:
                            # Text message
                            content = msg.content if msg.content else "[no text]"
                            line = f"[TEXT - {timestamp_str}] {msg.sender}: {content}"
                            transcript_lines.append(line)

                    full_transcript = "\n".join(transcript_lines)

                    # Estimate tokens for display
                    from utils.token_estimation import estimate_tokens, estimate_cost
                    token_count = estimate_tokens(full_transcript, config['llm_provider'])
                    status_text.markdown(
                        f"**✨ Sending full transcript to {config['llm_provider']}...**\n\n"
                        f"Tokens: ~{token_count:,}"
                    )

                    progress_bar.progress(75)

                    # Correct full transcript
                    corrected_transcript = corrector.correct_full_transcript(full_transcript)

                    # Parse corrected transcript back
                    status_text.markdown("**✨ Parsing corrected transcript...**")
                    corrected_lines = corrected_transcript.strip().split('\n')

                    # Match corrected lines back to original transcriptions
                    for line_idx, filename in filename_to_message.items():
                        if line_idx < len(corrected_lines):
                            corrected_line = corrected_lines[line_idx]

                            # Extract content after "Speaker: "
                            if ': ' in corrected_line:
                                corrected_content = corrected_line.split(': ', 1)[1]
                                trans_result = transcriptions[filename]
                                trans_result['corrected_text'] = corrected_content
                                trans_result['llm_corrected'] = True

                    # Estimate cost
                    if config['llm_provider'] != 'ollama':
                        # Get model name
                        if config['llm_provider'] == 'claude':
                            model = "claude-3-5-sonnet-20241022"
                        elif config['llm_provider'] == 'openai':
                            model = "gpt-4o"
                        elif config['llm_provider'] == 'mistral':
                            model = "mistral-large-latest"
                        else:
                            model = "unknown"

                        llm_correction_cost = estimate_cost(token_count, config['llm_provider'], model, mode='bulk')

                    status_text.markdown(
                        f"**✨ Bulk correction complete! Corrected {len(filename_to_message)} transcriptions at once**"
                    )

                else:
                    # Message-by-message mode (original behavior)
                    for idx, (filename, trans_result) in enumerate(transcriptions.items(), 1):
                        if trans_result['success'] and trans_result['text']:
                            progress = 70 + int((idx / len(transcriptions)) * 10)
                            progress_bar.progress(progress)

                            status_text.markdown(
                                f"**✨ Correcting {idx}/{len(transcriptions)}...**"
                            )

                            # Log LLM correction start
                            checkpoint_mgr.add_log_entry(checkpoint, "llm_correction_start", filename)

                            import time
                            llm_start_time = time.time()

                            corrected = corrector.correct_transcription(trans_result['text'])
                            trans_result['corrected_text'] = corrected
                            trans_result['llm_corrected'] = True

                            elapsed = time.time() - llm_start_time

                            # Log LLM correction complete
                            checkpoint_mgr.add_log_entry(checkpoint, "llm_correction_complete", filename, {
                                "duration": elapsed
                            })

                            # Save checkpoint periodically (every 5 corrections)
                            if idx % 5 == 0:
                                checkpoint_mgr.save_checkpoint(checkpoint, checkpoint_path)

                    # Final save after all corrections
                    checkpoint_mgr.save_checkpoint(checkpoint, checkpoint_path)

                    status_text.markdown(f"**✨ Applied AI corrections using {config['llm_provider']}**")

            except Exception as e:
                logger.error(f"LLM correction failed: {e}")
                results['errors'].append(f"AI correction failed: {e}")
                status_text.warning(f"⚠️ AI correction failed: {e}")

        progress_bar.progress(80)

        # Step 5: Generate timeline (80-90%)
        status_text.markdown("**📄 Generating timeline...**")

        # Save timeline(s)
        output_dir = tempfile.mkdtemp(prefix='whatsapp_output_')
        timeline_ext = 'md' if config['output_format'] == 'markdown' else config['output_format']

        # Build filename with custom base name, date range, and timestamp
        base_name = config.get('output_filename', 'WhatsApp_Chat')

        # Add date range if filtering is enabled
        if filter_stats.get('filtered'):
            start = filter_stats['start_date'].strftime('%Y%m%d')
            end = filter_stats['end_date'].strftime('%Y%m%d')
            date_range = f"_{start}-{end}"
        else:
            date_range = ""

        # Add timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Generate and save corrected version (or main version if no LLM used)
        if config.get('use_llm'):
            status_text.markdown("**📄 Generating AI-enhanced timeline...**")
            corrected_filename = f"{base_name}{date_range}_{timestamp}_corrected.{timeline_ext}"
        else:
            corrected_filename = f"{base_name}{date_range}_{timestamp}.{timeline_ext}"

        timeline_file = os.path.join(output_dir, corrected_filename)
        timeline_content = generate_timeline(
            messages, transcriptions, config['output_format'], config, filter_stats, use_corrected=True
        )
        # Use utf-8-sig (UTF-8 with BOM) on Windows for better compatibility with text editors
        encoding = 'utf-8-sig' if os.name == 'nt' else 'utf-8'
        with open(timeline_file, 'w', encoding=encoding) as f:
            f.write(timeline_content)

        results['timeline_file'] = timeline_file

        # Generate uncorrected version if requested and LLM was used
        if config.get('save_uncorrected', False) and config.get('use_llm'):
            status_text.markdown("**📄 Generating original timeline (uncorrected)...**")
            uncorrected_filename = f"{base_name}{date_range}_{timestamp}_uncorrected.{timeline_ext}"
            uncorrected_file = os.path.join(output_dir, uncorrected_filename)

            uncorrected_content = generate_timeline(
                messages, transcriptions, config['output_format'], config, filter_stats, use_corrected=False
            )
            # Use utf-8-sig (UTF-8 with BOM) on Windows for better compatibility
            encoding = 'utf-8-sig' if os.name == 'nt' else 'utf-8'
            with open(uncorrected_file, 'w', encoding=encoding) as f:
                f.write(uncorrected_content)

            # Add to results as a list
            results['timeline_file'] = [timeline_file, uncorrected_file]
            status_text.markdown("**✓ Generated both corrected and uncorrected timelines**")

        progress_bar.progress(90)

        # Step 6: Organize media (90-100%)
        status_text.markdown("**📁 Organizing media files...**")

        media_dir = os.path.join(output_dir, 'media')
        file_mapping = organizer.organize_media(media_dir)

        # Create media index
        index_file = os.path.join(media_dir, 'index.txt')
        organizer.create_media_index(file_mapping, index_file)

        # Create media ZIP with matching filename
        media_zip_filename = f"{base_name}{date_range}_{timestamp}_media.zip"
        media_zip_path = os.path.join(output_dir, media_zip_filename)
        create_output_zip(media_dir, media_zip_path)

        results['media_zip'] = media_zip_path
        progress_bar.progress(100)

        # Cleanup
        organizer.cleanup()
        os.unlink(zip_path)

        # Calculate statistics
        # Collect detected languages if auto-detect was used
        detected_languages = set()
        if config['language'] == 'auto':
            for trans in transcriptions.values():
                if trans.get('success') and trans.get('language'):
                    detected_languages.add(trans['language'])

        results['stats'] = {
            'total_messages': len(messages),
            'text_messages': parse_stats['text'],
            'voice_messages': len(audio_files),
            'transcribed': len([t for t in transcriptions.values() if t['success']]),
            'failed_transcriptions': len([t for t in transcriptions.values() if not t['success']]),
            'media_files': len(file_mapping),
            'transcription_engine': config['transcription_engine'],
            'transcription_cost': total_cost if config['transcription_engine'] == 'voxtral' else 0.0,
            'language_mode': config['language'],
            'detected_languages': ', '.join(sorted(detected_languages)) if detected_languages else None,
            'date_filtered': filter_stats['filtered'],
            'date_range': f"{filter_stats['start_date']} to {filter_stats['end_date']}" if filter_stats['filtered'] else None,
            'skipped_by_filter': filter_stats.get('skipped_messages', 0) + filter_stats.get('skipped_audio', 0) if filter_stats['filtered'] else 0,
            'llm_used': config['use_llm'],
            'llm_provider': config['llm_provider'] if config['use_llm'] else None,
            'llm_correction_mode': correction_mode if config['use_llm'] else None,
            'llm_correction_cost': llm_correction_cost if config['use_llm'] else 0.0
        }

        # Store filter stats for timeline generation
        results['filter_stats'] = filter_stats

        results['processing_time'] = (datetime.now() - start_time).total_seconds()
        results['success'] = True

        # Final checkpoint save with completion marker
        checkpoint['completed'] = True
        checkpoint_mgr.save_checkpoint(checkpoint, checkpoint_path)

        # Cleanup old checkpoints
        checkpoint_mgr.cleanup_old_checkpoints(days=7)

        # Include checkpoint path in results
        results['checkpoint_path'] = checkpoint_path

        status_text.markdown("**✅ Processing complete!**")

    except Exception as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        results['errors'].append(str(e))
        status_text.error(f"❌ Error: {e}")

    return results


def generate_timeline(messages, transcriptions, output_format, config, filter_stats, use_corrected=True):
    """
    Generate timeline in specified format.

    Args:
        messages: List of WhatsAppMessage objects
        transcriptions: Dictionary of transcriptions
        output_format: Output format (markdown, txt, csv, json)
        config: Configuration dictionary
        filter_stats: Date filter statistics
        use_corrected: Whether to use corrected text (True) or original only (False)

    Returns:
        Timeline content as string
    """
    if output_format == 'markdown':
        return generate_markdown_timeline(messages, transcriptions, config, filter_stats, use_corrected)
    elif output_format == 'txt':
        return generate_text_timeline(messages, transcriptions, config, filter_stats, use_corrected)
    elif output_format == 'csv':
        return generate_csv_timeline(messages, transcriptions, config, filter_stats, use_corrected)
    elif output_format == 'json':
        return generate_json_timeline(messages, transcriptions, config, filter_stats, use_corrected)
    else:
        raise ValueError(f"Unknown output format: {output_format}")


def generate_markdown_timeline(messages, transcriptions, config, filter_stats, use_corrected=True):
    """Generate timeline in Markdown format."""
    lines = []
    lines.append("# WhatsApp Chat Timeline\n\n")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    lines.append(f"**Total Messages:** {len(messages)}\n\n")

    # Add version info
    if not use_corrected:
        lines.append("**Version:** Original transcriptions only (uncorrected)\n\n")
    elif config.get('use_llm'):
        lines.append("**Version:** AI-enhanced transcriptions\n\n")

    # Add language info
    if config['language'] == 'auto':
        lines.append(f"**Language:** Auto-detect\n\n")
    else:
        lines.append(f"**Language:** {config['language'].upper()}\n\n")

    # Add date range filter info
    if filter_stats.get('filtered'):
        lines.append(f"**Date Range Filter:** {filter_stats['start_date']} to {filter_stats['end_date']}\n\n")
        lines.append(f"*Note: {filter_stats['skipped_messages']} messages and {filter_stats['skipped_audio']} audio files were excluded (outside date range)*\n\n")

    lines.append("---\n\n")

    for msg in messages:
        timestamp_str = msg.timestamp.strftime('%Y-%m-%d %H:%M:%S') if msg.timestamp else 'Unknown time'

        lines.append(f"## {timestamp_str}\n")
        lines.append(f"**{msg.sender}**\n\n")

        if msg.message_type == 'voice' and msg.media_file:
            # Check if we have transcription
            trans_result = transcriptions.get(msg.media_file, {})
            if trans_result.get('success'):
                # Use corrected text if requested and available, otherwise use original
                if use_corrected:
                    text = trans_result.get('corrected_text', trans_result.get('text', ''))
                else:
                    text = trans_result.get('text', '')
                lines.append(f"🎤 *Voice message:* {text}\n\n")

                if use_corrected and trans_result.get('llm_corrected'):
                    lines.append(f"*✨ AI-enhanced transcription*\n\n")
            else:
                lines.append(f"🎤 *Voice message* (transcription failed)\n\n")

        elif msg.message_type == 'text':
            lines.append(f"{msg.content}\n\n")

        else:
            # Media reference
            media_emoji = {
                'image': '📷',
                'video': '🎥',
                'document': '📄',
                'audio': '🎵'
            }.get(msg.message_type, '📎')

            lines.append(f"{media_emoji} *{msg.message_type.title()}*")
            if msg.media_file:
                lines.append(f": {msg.media_file}")
            lines.append("\n\n")

        lines.append("---\n\n")

    return ''.join(lines)


def generate_text_timeline(messages, transcriptions, config, filter_stats, use_corrected=True):
    """Generate timeline in plain text format."""
    lines = []
    lines.append("=" * 80 + "\n")
    lines.append("WhatsApp Chat Timeline\n")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"Total Messages: {len(messages)}\n")

    # Add version info
    if not use_corrected:
        lines.append("Version: Original transcriptions only (uncorrected)\n")
    elif config.get('use_llm'):
        lines.append("Version: AI-enhanced transcriptions\n")

    # Add language info
    if config['language'] == 'auto':
        lines.append(f"Language: Auto-detect\n")
    else:
        lines.append(f"Language: {config['language'].upper()}\n")

    # Add date range filter info
    if filter_stats.get('filtered'):
        lines.append(f"Date Range Filter: {filter_stats['start_date']} to {filter_stats['end_date']}\n")
        lines.append(f"Note: {filter_stats['skipped_messages']} messages and {filter_stats['skipped_audio']} audio files excluded\n")

    lines.append("=" * 80 + "\n\n")

    for msg in messages:
        timestamp_str = msg.timestamp.strftime('%Y-%m-%d %H:%M:%S') if msg.timestamp else 'Unknown time'

        lines.append(f"[{timestamp_str}] {msg.sender}\n")

        if msg.message_type == 'voice' and msg.media_file:
            trans_result = transcriptions.get(msg.media_file, {})
            if trans_result.get('success'):
                # Use corrected text if requested and available, otherwise use original
                if use_corrected:
                    text = trans_result.get('corrected_text', trans_result.get('text', ''))
                else:
                    text = trans_result.get('text', '')
                lines.append(f"[Voice message]: {text}\n")
            else:
                lines.append(f"[Voice message] (transcription failed)\n")

        elif msg.message_type == 'text':
            lines.append(f"{msg.content}\n")

        else:
            lines.append(f"[{msg.message_type.title()}]")
            if msg.media_file:
                lines.append(f": {msg.media_file}")
            lines.append("\n")

        lines.append("-" * 80 + "\n\n")

    return ''.join(lines)


def generate_csv_timeline(messages, transcriptions, config, filter_stats, use_corrected=True):
    """Generate timeline in CSV format."""
    import io
    output = io.StringIO()
    writer = csv.writer(output)

    # Add metadata as comments (CSV doesn't support headers well)
    lang_mode = 'Auto-detect' if config['language'] == 'auto' else config['language'].upper()
    output.write(f"# WhatsApp Chat Timeline\n")
    output.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Add version info
    if not use_corrected:
        output.write(f"# Version: Original transcriptions only (uncorrected)\n")
    elif config.get('use_llm'):
        output.write(f"# Version: AI-enhanced transcriptions\n")

    output.write(f"# Language: {lang_mode}\n")

    # Add date range filter info
    if filter_stats.get('filtered'):
        output.write(f"# Date Range Filter: {filter_stats['start_date']} to {filter_stats['end_date']}\n")
        output.write(f"# Note: {filter_stats['skipped_messages']} messages and {filter_stats['skipped_audio']} audio files excluded\n")

    # Header
    writer.writerow(['Timestamp', 'Sender', 'Type', 'Content', 'Media File', 'AI Enhanced', 'Language'])

    for msg in messages:
        timestamp_str = msg.timestamp.strftime('%Y-%m-%d %H:%M:%S') if msg.timestamp else ''

        content = msg.content
        ai_enhanced = 'No'
        detected_lang = ''

        if msg.message_type == 'voice' and msg.media_file:
            trans_result = transcriptions.get(msg.media_file, {})
            if trans_result.get('success'):
                # Use corrected text if requested and available, otherwise use original
                if use_corrected:
                    content = trans_result.get('corrected_text', trans_result.get('text', ''))
                else:
                    content = trans_result.get('text', '')
                ai_enhanced = 'Yes' if (use_corrected and trans_result.get('llm_corrected')) else 'No'
                detected_lang = trans_result.get('language', '')

        writer.writerow([
            timestamp_str,
            msg.sender,
            msg.message_type,
            content,
            msg.media_file or '',
            ai_enhanced,
            detected_lang
        ])

    return output.getvalue()


def generate_json_timeline(messages, transcriptions, config, filter_stats, use_corrected=True):
    """Generate timeline in JSON format."""
    metadata = {
        'generated': datetime.now().isoformat(),
        'total_messages': len(messages),
        'language_mode': config['language']
    }

    # Add version info
    if not use_corrected:
        metadata['version'] = 'Original transcriptions only (uncorrected)'
    elif config.get('use_llm'):
        metadata['version'] = 'AI-enhanced transcriptions'

    # Add date range filter info
    if filter_stats.get('filtered'):
        metadata['date_range_filter'] = {
            'start_date': str(filter_stats['start_date']),
            'end_date': str(filter_stats['end_date']),
            'skipped_messages': filter_stats['skipped_messages'],
            'skipped_audio': filter_stats['skipped_audio']
        }

    data = {
        'metadata': metadata,
        'messages': []
    }

    for msg in messages:
        msg_data = {
            'timestamp': msg.timestamp.isoformat() if msg.timestamp else None,
            'sender': msg.sender,
            'type': msg.message_type,
            'content': msg.content,
            'media_file': msg.media_file
        }

        if msg.message_type == 'voice' and msg.media_file:
            trans_result = transcriptions.get(msg.media_file, {})
            if trans_result.get('success'):
                if use_corrected:
                    # Include both versions for JSON
                    msg_data['transcription'] = {
                        'text': trans_result.get('text', ''),
                        'corrected_text': trans_result.get('corrected_text'),
                        'ai_enhanced': trans_result.get('llm_corrected', False),
                        'language': trans_result.get('language')
                    }
                else:
                    # Only original version
                    msg_data['transcription'] = {
                        'text': trans_result.get('text', ''),
                        'language': trans_result.get('language')
                    }

        data['messages'].append(msg_data)

    return json.dumps(data, indent=2, ensure_ascii=False)


def render_results(results):
    """Render results section."""
    st.header("✅ Processing Complete!")

    # Statistics card
    stats = results['stats']

    # Build language info string
    if stats['language_mode'] == 'auto':
        lang_info = f"Auto-detect ({stats['detected_languages']})" if stats['detected_languages'] else "Auto-detect"
    else:
        lang_info = stats['language_mode'].upper()

    # Build engine info string
    engine_display = {
        'faster-whisper': 'Faster-Whisper (Local)',
        'voxtral': 'Voxtral Mini (Mistral API)'
    }.get(stats['transcription_engine'], stats['transcription_engine'])

    # Build LLM correction info
    llm_correction_info = ""
    if stats['llm_used']:
        correction_mode_display = {
            'message': 'Message-by-Message',
            'bulk': 'Full Transcript (Bulk)'
        }.get(stats.get('llm_correction_mode', 'message'), stats.get('llm_correction_mode', 'message'))

        llm_correction_info = f"""
            <li><strong>AI Provider:</strong> {stats['llm_provider']}</li>
            <li><strong>Correction Mode:</strong> {correction_mode_display}</li>
            {f"<li><strong>AI Correction Cost:</strong> ${stats['llm_correction_cost']:.4f} USD</li>" if stats['llm_correction_cost'] > 0 else ""}
        """

    st.markdown(f"""
    <div class="stats-card">
        <h3>📊 Statistics</h3>
        <ul>
            <li><strong>Total Messages:</strong> {stats['total_messages']}</li>
            <li><strong>Text Messages:</strong> {stats['text_messages']}</li>
            <li><strong>Voice Messages:</strong> {stats['voice_messages']}</li>
            <li><strong>Successfully Transcribed:</strong> {stats['transcribed']}</li>
            <li><strong>Transcription Engine:</strong> {engine_display}</li>
            {f"<li><strong>Transcription Cost:</strong> ${stats['transcription_cost']:.4f} USD</li>" if stats['transcription_cost'] > 0 else ""}
            <li><strong>Language:</strong> {lang_info}</li>
            <li><strong>Media Files Organized:</strong> {stats['media_files']}</li>
            <li><strong>Processing Time:</strong> {results['processing_time']:.1f} seconds</li>
            {llm_correction_info}
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Download buttons
    st.subheader("📥 Downloads")

    # Handle both single file and dual file cases
    timeline_files = results['timeline_file']
    is_dual_timeline = isinstance(timeline_files, list)

    if is_dual_timeline:
        # Dual timeline case - show both files
        col1, col2, col3 = st.columns(3)

        with col1:
            if os.path.exists(timeline_files[0]):
                with open(timeline_files[0], 'rb') as f:
                    st.download_button(
                        label="📄 AI-Enhanced Timeline",
                        data=f.read(),
                        file_name=os.path.basename(timeline_files[0]),
                        mime='text/plain',
                        use_container_width=True
                    )

        with col2:
            if os.path.exists(timeline_files[1]):
                with open(timeline_files[1], 'rb') as f:
                    st.download_button(
                        label="📄 Original Timeline",
                        data=f.read(),
                        file_name=os.path.basename(timeline_files[1]),
                        mime='text/plain',
                        use_container_width=True
                    )

        with col3:
            if results['media_zip'] and os.path.exists(results['media_zip']):
                with open(results['media_zip'], 'rb') as f:
                    st.download_button(
                        label="📦 Media Files",
                        data=f.read(),
                        file_name=os.path.basename(results['media_zip']),
                        mime='application/zip',
                        use_container_width=True
                    )
    else:
        # Single timeline case
        col1, col2 = st.columns(2)

        with col1:
            if timeline_files and os.path.exists(timeline_files):
                with open(timeline_files, 'rb') as f:
                    st.download_button(
                        label="📄 Download Timeline",
                        data=f.read(),
                        file_name=os.path.basename(timeline_files),
                        mime='text/plain',
                        use_container_width=True
                    )

        with col2:
            if results['media_zip'] and os.path.exists(results['media_zip']):
                with open(results['media_zip'], 'rb') as f:
                    st.download_button(
                        label="📦 Download Media Files",
                        data=f.read(),
                        file_name=os.path.basename(results['media_zip']),
                        mime='application/zip',
                        use_container_width=True
                    )

    # Timeline preview - show first file (corrected version if dual)
    preview_file = timeline_files[0] if is_dual_timeline else timeline_files
    if preview_file and os.path.exists(preview_file):
        if is_dual_timeline:
            st.subheader("👁️ AI-Enhanced Timeline Preview (first 50 lines)")
        else:
            st.subheader("👁️ Timeline Preview (first 50 lines)")
        with open(preview_file, 'r', encoding='utf-8') as f:
            preview = ''.join(f.readlines()[:50])
        st.code(preview, language=None)

    # Process another chat button
    if st.button("🔄 Process Another Chat", use_container_width=True):
        st.session_state.processing_complete = False
        st.session_state.results = None
        st.rerun()


def main():
    """Main application."""
    initialize_session_state()

    # Title
    st.title("💬 WhatsApp Transcriber")
    st.markdown("Process WhatsApp chat exports and create comprehensive timelines with transcriptions")

    # Render sidebar
    render_sidebar()

    # Show results if processing is complete
    if st.session_state.processing_complete and st.session_state.results:
        render_results(st.session_state.results)
        return

    # Upload section
    uploaded_file = render_upload_section()

    # Parse uploaded file for date analysis
    date_filter = None
    if uploaded_file:
        # Check if we need to parse the file
        if st.session_state.current_file_name != uploaded_file.name:
            # New file uploaded, parse it
            with st.spinner("Analyzing uploaded file..."):
                try:
                    # Save and extract
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        zip_path = tmp_file.name

                    organizer = FileOrganizer()
                    temp_dir = organizer.extract_zip(zip_path)

                    # Parse chat file
                    chat_file = organizer.get_chat_file()
                    if not chat_file:
                        st.error("❌ Could not find WhatsApp chat file in ZIP")
                        st.session_state.uploaded_data = None
                        st.session_state.current_file_name = None
                    else:
                        parser = WhatsAppParser()
                        messages = parser.parse_file(chat_file)
                        audio_files = organizer.get_audio_files()

                        # Analyze date range
                        date_info = analyze_date_range(messages, audio_files)

                        # Store in session state
                        st.session_state.uploaded_data = {
                            'messages': messages,
                            'audio_files': audio_files,
                            'date_info': date_info,
                            'organizer': organizer,
                            'zip_path': zip_path
                        }
                        st.session_state.current_file_name = uploaded_file.name

                        # Check for existing checkpoint
                        from utils import CheckpointManager
                        checkpoint_mgr = CheckpointManager()

                        # Generate conversation ID
                        conv_name = extract_conversation_name(uploaded_file.name)
                        # Use "all" for date range in checkpoint ID to make it more stable
                        # (actual date range is stored in checkpoint metadata)
                        conversation_id = checkpoint_mgr.generate_conversation_id(
                            conv_name,
                            "all",  # Changed from date range to "all" for stability
                            len(audio_files)
                        )

                        # Debug logging
                        logger.info(f"Checkpoint detection: conv_name='{conv_name}', files={len(audio_files)}, id={conversation_id}")

                        # Look for existing checkpoint
                        checkpoint_path = checkpoint_mgr.find_checkpoint_for_chat(conversation_id)
                        logger.info(f"Exact ID match result: {checkpoint_path}")

                        # Fallback: If exact match not found, search for similar checkpoints
                        if not checkpoint_path:
                            import glob
                            checkpoint_dir = checkpoint_mgr.checkpoint_dir
                            all_checkpoints = glob.glob(os.path.join(checkpoint_dir, "*_checkpoint.json"))
                            logger.info(f"Exact match not found. Searching {len(all_checkpoints)} checkpoint(s) by file count...")

                            # Look for checkpoints with matching file count (most reliable indicator)
                            for cp_path in all_checkpoints:
                                try:
                                    from utils import load_checkpoint
                                    cp = load_checkpoint(cp_path)
                                    logger.info(f"Checking {os.path.basename(cp_path)}: {cp['total_files']} files, name='{cp.get('chat_name')}'")

                                    # Match by file count (exact) - most reliable indicator
                                    if cp['total_files'] == len(audio_files):
                                        checkpoint_path = cp_path
                                        st.info(f"📋 Found existing checkpoint by file count match: {os.path.basename(cp_path)}")
                                        logger.info(f"✓ Found matching checkpoint by file count: {cp_path}")
                                        break
                                except Exception as e:
                                    logger.warning(f"Could not load checkpoint {cp_path}: {e}")
                                    continue

                            if not checkpoint_path:
                                logger.info("No matching checkpoint found by fallback search")

                        if checkpoint_path:
                            st.session_state.checkpoint_found = True
                            st.session_state.checkpoint_path = checkpoint_path
                        else:
                            st.session_state.checkpoint_found = False
                            st.session_state.checkpoint_path = None

                    # Clean up temporary file
                    os.unlink(zip_path)

                except Exception as e:
                    st.error(f"❌ Error analyzing file: {e}")
                    st.session_state.uploaded_data = None
                    st.session_state.current_file_name = None

        # Show checkpoint resume UI if found
        if st.session_state.checkpoint_found and st.session_state.checkpoint_path:
            st.divider()
            st.warning("⚠️ Previous processing found for this chat")

            from utils import load_checkpoint
            checkpoint = load_checkpoint(st.session_state.checkpoint_path)

            col1, col2, col3 = st.columns(3)
            col1.metric("Processed", f"{checkpoint['processed_files']}/{checkpoint['total_files']}")
            col2.metric("Failed", checkpoint['failed_files'])
            last_run = checkpoint['timestamp'][:16] if len(checkpoint['timestamp']) >= 16 else checkpoint['timestamp']
            col3.metric("Last Run", last_run)

            st.caption(f"💡 Chat: {checkpoint['chat_name']}")

            # Primary actions row
            col_a, col_b, col_c = st.columns(3)

            if col_a.button("▶️ Resume Processing", use_container_width=True):
                st.session_state.resume_checkpoint = checkpoint
                st.session_state.processing_mode = 'resume'
                st.rerun()

            if col_b.button("🔄 Start Fresh", use_container_width=True):
                os.remove(st.session_state.checkpoint_path)
                st.session_state.checkpoint_found = False
                st.session_state.checkpoint_path = None
                st.session_state.processing_mode = 'fresh'
                st.success("✓ Checkpoint deleted. Processing will start fresh.")
                st.rerun()

            # Show retry button only if there are failed files
            if checkpoint['failed_files'] > 0:
                if col_c.button("🔁 Retry Failed Files", use_container_width=True):
                    st.session_state.resume_checkpoint = checkpoint
                    st.session_state.processing_mode = 'retry'
                    st.rerun()
            else:
                col_c.button("✓ No Failed Files", use_container_width=True, disabled=True)

            # Secondary actions row
            col_d, col_e = st.columns(2)

            # Export processing log button
            if col_d.button("📊 Export Processing Log", use_container_width=True):
                from utils import CheckpointManager
                checkpoint_mgr = CheckpointManager()
                log_csv = checkpoint_mgr.export_processing_log(checkpoint)

                st.download_button(
                    label="💾 Download Log CSV",
                    data=log_csv,
                    file_name=f"{checkpoint['chat_name']}_processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            if col_e.button("📄 View Log Summary", use_container_width=True):
                # Show log summary in expander
                with st.expander("📋 Processing Log Details", expanded=True):
                    st.markdown("### Summary Statistics")
                    summary_cols = st.columns(4)
                    summary_cols[0].metric("Success Rate", f"{((checkpoint['processed_files'] - checkpoint['failed_files']) / checkpoint['total_files'] * 100):.1f}%")
                    summary_cols[1].metric("Total Cost", f"${checkpoint['stats']['total_transcription_cost'] + checkpoint['stats']['total_llm_cost']:.4f}")
                    summary_cols[2].metric("Duration", f"{checkpoint['stats']['total_duration_minutes']:.1f}m")
                    summary_cols[3].metric("Events", len(checkpoint.get('processing_log', [])))

                    st.markdown("### Recent Events")
                    recent_events = checkpoint.get('processing_log', [])[-10:]
                    if recent_events:
                        for event in reversed(recent_events):
                            event_type = event.get('event', 'unknown')
                            event_icon = {
                                'transcription_start': '▶️',
                                'transcription_complete': '✅',
                                'transcription_failed': '❌',
                                'llm_correction_start': '✨',
                                'llm_correction_complete': '✨',
                                'error': '⚠️'
                            }.get(event_type, '📝')
                            st.text(f"{event_icon} {event.get('time', '')[:19]} - {event_type} - {event.get('file', '')}")
                    else:
                        st.info("No events logged yet")

        # Show date range filter if data is available
        if st.session_state.uploaded_data:
            data = st.session_state.uploaded_data
            date_filter = render_date_range_filter(
                data['date_info'],
                data['messages'],
                data['audio_files']
            )

    # Configuration section
    config = render_configuration_section()

    # Token estimation and cost display (for bulk correction mode)
    if uploaded_file and st.session_state.uploaded_data and config['use_llm'] and config['correction_mode'] == 'bulk':
        from utils.token_estimation import estimate_tokens, format_token_count, estimate_cost, check_token_limit

        st.divider()
        st.subheader("📊 Token Estimation (Bulk Mode)")

        # Get the data
        messages = st.session_state.uploaded_data['messages']

        # Apply date filtering if enabled
        if date_filter and date_filter.get('enabled'):
            start_date = date_filter['start_date']
            end_date = date_filter['end_date']
            messages = [msg for msg in messages if msg.timestamp and start_date <= msg.timestamp.date() <= end_date]

        # Build a rough transcript to estimate size
        # Format: [Type - Date] Speaker: Content
        transcript_lines = []
        for msg in messages:
            timestamp_str = msg.timestamp.strftime("%Y-%m-%d %H:%M:%S") if msg.timestamp else "Unknown"
            msg_type = "TRANSCRIPTION" if msg.message_type == 'voice' else "TEXT"
            # For audio messages, use a placeholder text of ~50 chars as estimate
            content = msg.content if msg.content else "[transcription placeholder - approximately 50 characters]"
            line = f"[{msg_type} - {timestamp_str}] {msg.sender}: {content}"
            transcript_lines.append(line)

        rough_transcript = "\n".join(transcript_lines)

        # Estimate tokens
        provider = config['llm_provider']
        token_count = estimate_tokens(rough_transcript, provider)

        # Determine which model will be used
        if provider == 'claude':
            model = "claude-3-5-sonnet-20241022"
        elif provider == 'openai':
            model = "gpt-4o"
        elif provider == 'mistral':
            model = "mistral-large-latest"
        else:
            model = config.get('ollama_model', 'llama3.2')

        # Check token limits
        limit_check = check_token_limit(token_count, provider, model)

        # Display results
        col_a, col_b = st.columns(2)

        with col_a:
            st.metric("Estimated Tokens", format_token_count(token_count))

        with col_b:
            if provider != 'ollama':
                estimated_cost = estimate_cost(token_count, provider, model, mode='bulk')
                st.metric("Estimated Cost", f"${estimated_cost:.4f}")
            else:
                st.metric("Estimated Cost", "FREE (Local)")

        # Show limit warnings
        if limit_check['error']:
            st.error(limit_check['message'])
            st.warning("⚠️ Switch to Message-by-Message mode to process this conversation.")
        elif limit_check['warning']:
            st.warning(limit_check['message'])
        else:
            st.success(limit_check['message'])

        # Show additional info
        st.caption(f"ℹ️ This is a rough estimate based on {len(messages)} messages. Actual token count may vary.")
        st.caption(f"ℹ️ Model: {model}")

    # Process button
    st.divider()

    if uploaded_file and st.session_state.uploaded_data:
        # Check date filter validity
        if date_filter and date_filter.get('enabled') and not date_filter.get('valid', True):
            st.error("❌ Please fix the date range before processing")
        else:
            process_button = st.button(
                "🚀 Process Chat",
                type="primary",
                use_container_width=True
            )

            if process_button:
                # Add date filter to config
                config['date_filter'] = date_filter

                # Validate configuration
                if config['use_llm']:
                    if config['llm_provider'] in ['claude', 'openai', 'mistral'] and not config['llm_api_key']:
                        st.error(f"❌ API key required for {config['llm_provider'].title()}")
                        return

                # Validate Mistral API key for Voxtral
                if config['transcription_engine'] == 'voxtral' and not config['mistral_api_key']:
                    st.error("❌ Mistral API key required for Voxtral transcription")
                    return

                # Process
                with st.container():
                    st.subheader("⚙️ Processing")

                    # Check if resuming from checkpoint or retrying failed files
                    resume_checkpoint = st.session_state.get('resume_checkpoint')
                    processing_mode = st.session_state.get('processing_mode')

                    if processing_mode == 'resume' and resume_checkpoint:
                        st.info(f"▶️ Resuming from checkpoint: {resume_checkpoint['processed_files']}/{resume_checkpoint['total_files']} files completed")
                        results = process_chat(uploaded_file, config, checkpoint=resume_checkpoint)
                        # Clear resume state
                        st.session_state.resume_checkpoint = None
                        st.session_state.processing_mode = None
                    elif processing_mode == 'retry' and resume_checkpoint:
                        st.info(f"🔁 Retrying {resume_checkpoint['failed_files']} failed files")
                        results = process_chat(uploaded_file, config, checkpoint=resume_checkpoint, retry_mode=True)
                        # Clear retry state
                        st.session_state.resume_checkpoint = None
                        st.session_state.processing_mode = None
                    else:
                        results = process_chat(uploaded_file, config)

                    if results['success']:
                        st.session_state.processing_complete = True
                        st.session_state.results = results
                        # Clear uploaded data
                        st.session_state.uploaded_data = None
                        st.session_state.current_file_name = None
                        st.rerun()
                    else:
                        st.error("❌ Processing failed")
                        for error in results['errors']:
                            st.error(error)
    elif uploaded_file:
        st.info("👆 Analyzing uploaded file...")
    else:
        st.info("👆 Please upload a WhatsApp export ZIP file to begin")

    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 14px;'>
        <p>WhatsApp Transcriber | Free & Open Source</p>
        <p>Your data is processed locally and never sent to external servers (except when using cloud AI providers)</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
