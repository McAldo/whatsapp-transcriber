# WhatsApp Transcriber =�

A powerful Python Streamlit application that processes WhatsApp chat exports and creates comprehensive timelines with voice message transcriptions. Everything runs locally on your computer with mobile access support!

## ( Features

- **Local Voice Transcription** - Uses faster-whisper to transcribe voice messages completely offline (no API key needed)
- **Cloud Transcription (Optional)** - Voxtral Mini API for superior quality, especially for Italian (~$0.001/minute)
- **Checkpoint & Resume** - Automatically saves progress! Resume interrupted processing without losing work or re-processing files
- **AI Enhancement (Optional)** - Improve transcriptions with Claude, OpenAI, Mistral AI, or Ollama (local & free)
- **Smart Correction Modes** - Choose between Message-by-Message (reliable) or Full Transcript (faster, cheaper)
- **Token Estimation** - See estimated costs before processing with automatic limit checking
- **Date Range Filtering** - Process specific date ranges from your conversations
- **Mobile-Friendly** - Access the app from your phone on the same WiFi network
- **Multiple Output Formats** - Export timelines as Markdown, plain text, CSV, or JSON
- **Complete Timeline** - Combines text messages, transcribed voice messages, and media references chronologically
- **Media Organization** - Automatically organizes and renames all media files with timestamps
- **Privacy First** - Everything runs locally by default (except when using cloud AI providers)
- **UTF-8 Encoding** - Proper support for all languages including Italian accents and special characters

## 📥 Installation from GitHub

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Git (to clone repository)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/whatsapp-transcriber.git
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

4. **Configure environment variables** (optional - for AI features)
   ```bash
   # Copy template
   cp .env.example .env

   # Edit .env and add your API keys
   # Use any text editor
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Access the app**
   - Local: `http://localhost:8501`
   - Mobile: `http://YOUR_COMPUTER_IP:8501`

## 🚀 Quick Start

If you already have the files locally, here's the quick version:

1. **Install dependencies:** `pip install -r requirements.txt`
2. **Run the app:** `streamlit run app.py`
3. **Open browser:** `http://localhost:8501`

## =� Access from Mobile

### Connect Your Phone

1. Make sure your phone is on the **same WiFi network** as your computer
2. Find your computer's local IP address:

**Windows:**
```cmd
ipconfig
```
Look for "IPv4 Address" (e.g., 192.168.1.100)

**Mac:**
```bash
ifconfig | grep "inet "
```

**Linux:**
```bash
hostname -I
```

3. On your phone, open a browser and go to:
```
http://[YOUR-COMPUTER-IP]:8501
```
Example: `http://192.168.1.100:8501`

### Troubleshooting Mobile Access

- **Can't connect?** Check your firewall settings - port 8501 must be open
- **Windows Firewall:** Allow Python/Streamlit through Windows Defender Firewall
- **Mac Firewall:** System Preferences � Security & Privacy � Firewall Options
- **Linux:** Check iptables rules

## =� How to Use

### Step 1: Export WhatsApp Chat

**On Android:**
1. Open WhatsApp chat
2. Tap the three dots (�) � More � Export chat
3. Choose "Include Media"
4. Save the ZIP file

**On iPhone:**
1. Open WhatsApp chat
2. Tap contact/group name at top
3. Scroll down � Export Chat
4. Choose "Attach Media"
5. Save the ZIP file

### Step 2: Process the Chat

1. Upload the ZIP file to the app
2. Configure settings:
   - **Transcription Model:** Choose based on speed vs. quality
     - `base`: Fast, good quality (recommended)
     - `small`: Balanced
     - `medium/large`: Best quality, slower
   - **Output Format:** Choose timeline format (Markdown recommended)
   - **AI Enhancement (Optional):** Enable to improve transcriptions

3. Click "Process Chat" and wait
4. Download your timeline and organized media files!


## 🎤 Transcription Engines

Choose between two transcription engines based on your needs:

### Faster-Whisper (Default - Local & Free)

**Best for:** Most users, privacy-focused, offline transcription

- ✅ **Runs locally** on your computer
- ✅ **100% free** - no API costs
- ✅ **Works offline** - no internet needed
- ✅ **Good quality** for most languages
- ✅ **GPU acceleration** supported (5-10x faster)
- ℹ️ **Privacy:** Everything stays on your machine

**Setup:** Already included! Just select your model size and start processing.

### Voxtral Mini (Mistral API - Cloud-Based)

**Best for:** Italian transcription, highest quality needs

- ✅ **Better quality** than Whisper, especially for Italian
- ✅ **Fast** API-based processing
- ✅ **Very affordable** - ~$0.001/minute (~$0.06/hour)
- ⚠️ **Requires internet** connection
- ⚠️ **Requires API key** from Mistral
- ℹ️ **Privacy:** Audio files sent to Mistral's API

**Cost Examples:**
- 10-minute conversation: ~$0.01
- 1-hour conversation: ~$0.06
- 10 hours of audio: ~$0.60

**Setup:**

1. **Get Mistral API Key:**
   - Sign up at [console.mistral.ai](https://console.mistral.ai/)
   - Navigate to API Keys section
   - Create a new API key
   - New users get free credits to try the service!

2. **Add API Key:**

   **Option A - Environment Variable (Recommended):**
   ```bash
   # Create .env file in project directory
   echo "MISTRAL_API_KEY=your_key_here" >> .env
   ```

   **Option B - Manual Entry:**
   - Enter API key in the app UI when using Voxtral

3. **Select Voxtral in App:**
   - Choose "Voxtral Mini (Mistral API)" as transcription engine
   - The app will show estimated cost before processing
   - Test connection to verify API key works

### Which Engine Should I Choose?

| Feature | Faster-Whisper | Voxtral Mini API |
|---------|----------------|------------------|
| **Quality** | Good | Better |
| **Italian** | Good | Excellent ⭐ |
| **Speed** | Fast (with GPU) | Fast |
| **Cost** | Free | ~$0.001/min |
| **Privacy** | 100% local | Sends to API |
| **Internet** | Not required | Required |
| **Setup** | None | Need API key |

**Recommendation:**
- **For most users:** Start with Faster-Whisper (free, private, works great)
- **For Italian chats:** Try Voxtral Mini for superior quality
- **For highest quality:** Use Voxtral Mini (very affordable)
- **For privacy/offline:** Use Faster-Whisper only

## > AI Enhancement Options

### Option 1: Ollama (Recommended - FREE & PRIVATE)

**Best for privacy** - Everything stays on your computer!

1. **Install Ollama:**
   - Download from [ollama.ai](https://ollama.ai)
   - Run the installer

2. **Pull a model:**
```bash
ollama pull llama3.2
```

3. **Start Ollama** (usually starts automatically)

4. **In the app:**
   - Enable "AI correction"
   - Select "Ollama (Local)"
   - Choose your model (llama3.2 recommended)
   - Click "Test Connection"

**Recommended models:**
- `llama3.2` - Lightweight, fast, good quality
- `llama3.1` - Larger, better quality
- `mistral` - Good alternative

## 🔄 Checkpoint & Resume System

**Never lose progress again!** The app automatically saves your work as it processes.

### How It Works

- **Automatic Checkpoints** - Progress saved after every file transcribed
- **Smart Resume** - Interrupted? Just restart and resume where you left off
- **No Re-Processing** - Already completed files are skipped automatically
- **Settings Preserved** - Your transcription engine, LLM settings, and preferences are restored
- **Cost Tracking** - Keeps track of API costs across resume sessions

### Using Resume

1. **Start Processing** - Upload your chat and click "Process Chat"
2. **If Interrupted** - App crashes, computer restarts, or you close it - no problem!
3. **Restart & Upload** - Start the app again and upload the SAME ZIP file
4. **Resume Button Appears** - You'll see "⚠️ Previous processing found for this chat"
5. **Click Resume** - Processing continues from where it stopped!

**Checkpoints stored in:** `chat_checkpoints/` directory

**Debug tools included:**
- `check_checkpoint.py` - View checkpoint status and statistics
- `repair_checkpoint_encoding.py` - Fix encoding issues in existing checkpoints

### Option 2: Claude API

1. Get API key from [console.anthropic.com](https://console.anthropic.com)

2. **Option A - Environment Variable (Recommended):**
   - Create `.env` file in project directory
   - Add: `ANTHROPIC_API_KEY=your_key_here`

3. **Option B - Manual Entry:**
   - Enter API key in the app UI

### Option 3: OpenAI API

1. Get API key from [platform.openai.com](https://platform.openai.com)

2. **Option A - Environment Variable (Recommended):**
   - Create `.env` file in project directory
   - Add: `OPENAI_API_KEY=your_key_here`

3. **Option B - Manual Entry:**
   - Enter API key in the app UI

### Option 4: Mistral AI

1. Get API key from [console.mistral.ai](https://console.mistral.ai)

2. **Option A - Environment Variable (Recommended):**
   - Create `.env` file in project directory
   - Add: `MISTRAL_API_KEY=your_key_here`

3. **Option B - Manual Entry:**
   - Enter API key in the app UI

## 🔄 LLM Correction Modes

When using AI correction, you can choose between two processing modes:

### Message-by-Message Mode (Default)

**How it works:**
- Each voice message transcription is sent individually to the AI
- AI corrects one transcription at a time
- Safe for any conversation size

**Best for:**
- Long conversations (100+ messages)
- When using smaller AI models
- When you want reliable processing

**Pros:**
- ✅ Works with any conversation size
- ✅ No token limit concerns
- ✅ More predictable costs

**Cons:**
- ⏱️ Slower processing (one API call per message)
- 💰 More expensive for large conversations (many API calls)

---

### Full Transcript Mode (Bulk)

**How it works:**
- Entire conversation sent as one transcript to the AI
- AI corrects all transcriptions in a single pass
- Understands conversation context

**Best for:**
- Small to medium conversations (< 50 messages)
- When speed matters
- Cost optimization

**Pros:**
- ⚡ Much faster (single API call)
- 💰 Cheaper for conversations with many voice messages
- 🎯 Better context understanding (AI sees full conversation)

**Cons:**
- ⚠️ Requires large context window (100k+ tokens)
- ⚠️ May fail on very long conversations
- ⚠️ Not suitable for models with small context limits

---

### Comparison Table

| Feature | Message-by-Message | Full Transcript |
|---------|-------------------|-----------------|
| **Speed** | Slower | Much Faster |
| **Cost** (50 msgs) | Higher | Lower |
| **Context Window** | Any model | 100k+ tokens needed |
| **Conversation Size** | Unlimited | Limited by tokens |
| **Context Awareness** | Per message only | Full conversation |
| **Reliability** | Very reliable | May fail on large chats |

### Which Models Support Full Transcript Mode?

**✅ Large Context Models (Recommended for Bulk):**
- **Claude 3.5 Sonnet** - 200k tokens (best for bulk)
- **GPT-4o** - 128k tokens (great for bulk)
- **Mistral Large** - 128k tokens (great for bulk)

**⚠️ Limited Context Models (Message-by-Message Only):**
- **GPT-4o-mini** - 128k tokens (can work for small chats)
- **Mistral Small** - 32k tokens (message-by-message recommended)
- **Most Ollama models** - 4k-32k tokens (message-by-message only)

### Token Estimation

Before processing in Bulk mode, the app will:
- ✅ Estimate total token count
- ✅ Show estimated cost
- ✅ Warn if conversation is too large
- ✅ Suggest switching to Message-by-Message if needed

**Typical token counts:**
- 10 messages with voice: ~2,000 tokens
- 50 messages with voice: ~10,000 tokens
- 100 messages with voice: ~20,000 tokens

### Cost Examples (Approximate)

**10 voice messages:**
- Message-by-Message: ~$0.02 (OpenAI GPT-4o-mini)
- Full Transcript: ~$0.01 (OpenAI GPT-4o)
- **Savings: 50%**

**50 voice messages:**
- Message-by-Message: ~$0.10
- Full Transcript: ~$0.03
- **Savings: 70%**

**Note:** Ollama (local) is always free for both modes!

### Recommendations

**Use Message-by-Message when:**
- 🔹 Conversation has 100+ messages
- 🔹 Using Ollama or models with small context windows
- 🔹 You want guaranteed reliability
- 🔹 Cost is not a primary concern

**Use Full Transcript when:**
- 🔹 Conversation has < 50 messages
- 🔹 Using Claude, GPT-4o, or Mistral Large
- 🔹 Speed is important
- 🔹 You want to minimize API costs
- 🔹 Token estimate shows "within limits"

## =� Output Files

### Timeline File

Contains chronologically ordered messages:
- Text messages with original timestamps
- Transcribed voice messages
- References to media files (images, videos, documents)

Available formats:
- **Markdown (.md)** - Most readable, formatted
- **Plain Text (.txt)** - Simple text format
- **CSV (.csv)** - For spreadsheet import
- **JSON (.json)** - For programmatic processing

### Media Files (ZIP)

Contains:
- All media files from the chat
- Renamed with timestamps: `YYYYMMDD_HHMMSS_filename`
- `index.txt` - List of all files with original names

## � Configuration

### Model Sizes

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| tiny | ~75 MB | Very Fast | Basic | Testing, clear audio |
| base | ~150 MB | Fast | Good | Quick processing (recommended) |
| small | ~500 MB | Medium | Very Good | Best balance |
| medium | ~1.5 GB | Slow | Excellent | Important transcriptions |
| large | ~3 GB | Very Slow | Best | Critical accuracy |

### Model Download Behavior

**Important:** Whisper models are downloaded automatically on first use!

- Models are downloaded when you first process a chat (not when selected in UI)
- Download happens only once - models are cached for future use
- Cache location: `~/.cache/huggingface/hub/`
- The app will show a status message during download
- **Be patient on first run** - downloads can take several minutes depending on your internet speed

**Download times (approximate):**
- tiny: ~1 minute
- base: ~2-3 minutes
- small: ~5-10 minutes
- medium: ~10-20 minutes
- large: ~20-30+ minutes

**Note:** The UI will indicate if a model is already downloaded (green checkmark) or needs to be downloaded (blue info message).

### Performance Tips

- **GPU acceleration:** Faster-whisper will use CUDA if available (significantly faster)
- **Large chats:** May take several minutes depending on voice message count
- **Memory:** Large models require more RAM (4GB+ recommended for medium/large)
- **Multiple chats:** Once a model is downloaded, subsequent uses are much faster

## =� Troubleshooting

### Installation Issues

**faster-whisper installation fails:**
```bash
# Try installing with specific versions
pip install faster-whisper==0.10.0 --upgrade
```

**On Windows - Microsoft Visual C++ required:**
- Download and install [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)

### Ollama Issues

**Connection refused:**
1. Check if Ollama is running:
```bash
ollama list
```
2. If not running, start it from the Ollama app

**Model not found:**
```bash
ollama pull llama3.2
```

**Port conflict:**
- Ollama uses port 11434 by default
- Change URL in app if using different port

### GPU Acceleration Issues

**Error: "Could not locate cudnn_ops64_9.dll" or "Cannot load symbol cudnnCreateTensorDescriptor"**

This means **cuDNN (CUDA Deep Neural Network library) is missing**. This is the most common GPU issue.

**Solution - Install cuDNN (EASIEST METHOD):**

**Option 1: Install via pip (Recommended - Works on Windows/Linux):**

First, check your CUDA version:
```bash
nvidia-smi
```
Look for "CUDA Version: 12.x" or "11.x"

Then install matching cuDNN:

```bash
# For CUDA 12.x (most common)
pip install nvidia-cudnn-cu12

# For CUDA 11.x
pip install nvidia-cudnn-cu11
```

**Important:** After installing, you need to add cuDNN to your PATH:

**Windows:**
```bash
# Find where pip installed it (run this):
python -c "import nvidia.cudnn; import os; print(os.path.dirname(nvidia.cudnn.__file__))"

# This will show a path like:
# C:\Users\YourName\AppData\Local\Programs\Python\Python311\Lib\site-packages\nvidia\cudnn\bin

# Add that bin folder to your PATH:
# 1. Search "Environment Variables" in Windows
# 2. Edit "Path" in System Variables
# 3. Click "New" and add the path shown above
# 4. Click OK and restart your terminal
```

**Linux:**
```bash
# Add to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(python -c "import nvidia.cudnn; import os; print(os.path.dirname(nvidia.cudnn.__file__))")/lib:$LD_LIBRARY_PATH

# Make it permanent (add to ~/.bashrc):
echo 'export LD_LIBRARY_PATH=$(python -c "import nvidia.cudnn; import os; print(os.path.dirname(nvidia.cudnn.__file__))")/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
```

**Verify installation:**
```bash
# This should work without errors:
python -c "import nvidia.cudnn; print(nvidia.cudnn.__version__)"
```

**Restart your terminal and the app** - GPU should now work!

---

**Troubleshooting pip install:**

If `pip install nvidia-cudnn-cu12` fails with version/dependency errors:

```bash
# Try specifying a compatible version:
pip install nvidia-cudnn-cu12==9.1.0.70

# Or for CUDA 11:
pip install nvidia-cudnn-cu11==8.9.7.29

# Check available versions:
pip index versions nvidia-cudnn-cu12
```

If pip installation keeps failing, use Option 3 (manual download) below.

---

**Option 2: Linux package manager (Ubuntu/Debian):**
```bash
# For CUDA 12.x
sudo apt-get install libcudnn9-cuda-12

# For CUDA 11.x
sudo apt-get install libcudnn8-cuda-11
```

---

**Option 3: Manual download (if pip fails):**

Some systems may need manual installation:

1. Go to [NVIDIA cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive)
2. Sign in (free NVIDIA Developer account required)
3. Download the version matching your CUDA (check with `nvidia-smi`)
4. Extract and follow platform-specific instructions in the archive

---

**GPU detected but falls back to CPU (other reasons):**

The app automatically tries multiple compute types:
1. `float16` (fastest, modern GPUs)
2. `int8_float16` (mixed precision)
3. `int8` (works on all GPUs)
4. Falls back to CPU if all fail

**Other common causes:**
- Older GPU architecture (needs compute capability 3.5+)
- CTranslate2 installed without CUDA support
- CUDA drivers not properly installed

**Fix: Reinstall CTranslate2 with GPU support:**
```bash
pip uninstall ctranslate2
pip install ctranslate2 --extra-index-url https://pypi.nvidia.com
```

**Verify CUDA is working:**
```bash
nvidia-smi
```
Should show your GPU and CUDA version.

**If still not working:**
- Update NVIDIA drivers from [nvidia.com](https://www.nvidia.com/Download/index.aspx)
- Reinstall CUDA Toolkit: [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
- Check GPU compute capability: [NVIDIA GPU List](https://developer.nvidia.com/cuda-gpus)

**Note:** The app will still work on CPU, just slower. GPU acceleration provides 5-10x speedup.

### Whisper Model Download Issues

**Model download fails or times out:**
1. Check your internet connection
2. Try a smaller model first (base or tiny)
3. Manually download model:
```bash
# The model will download automatically on first use in Python
python -c "from faster_whisper import WhisperModel; WhisperModel('base')"
```

**Download stuck or appears frozen:**
- The app IS working - downloads are large and take time
- Check the status message - it will update when download completes
- Don't close the app during download
- Check your HuggingFace cache: `~/.cache/huggingface/hub/`

**Where are models stored?**
- **Windows:** `C:\Users\YourName\.cache\huggingface\hub\`
- **Mac/Linux:** `~/.cache/huggingface/hub/`
- You can delete cached models to free space

### Processing Errors

**"Could not find chat text file":**
- Ensure you exported the chat correctly with WhatsApp's export feature
- The ZIP must contain the `_chat.txt` or similar file

**"Out of memory":**
- Try a smaller Whisper model (base or tiny)
- Close other applications
- Process smaller chats separately

**Transcription quality poor:**
- Try a larger Whisper model (small or medium)
- Enable AI correction with Ollama/Claude/OpenAI
- Check audio quality in original messages

**Characters showing as � or \u00XX (encoding issues):**
- The app now uses UTF-8-BOM encoding for Windows compatibility
- Italian accents (è, à, ò, ù) should display correctly
- Open files in Notepad++ or any UTF-8 capable editor
- If issues persist, run `python repair_checkpoint_encoding.py`

**Processing interrupted/crashed:**
- Don't worry! Your progress is saved
- Restart the app
- Upload the same ZIP file
- Click "Resume Processing" when it appears
- Processing continues from where it stopped

### Mobile Access Issues

**Phone can't connect:**
1. Verify both devices on same WiFi
2. Check computer's IP address is correct
3. Disable VPN on computer
4. Check firewall settings
5. Try accessing from computer first to ensure app is running

**Slow on mobile:**
- Normal - processing happens on computer
- Don't close the mobile browser while processing
- Check WiFi signal strength

## = Privacy & Security

### Data Processing

- **Local by Default**: Voice transcription happens on your computer using faster-whisper. No data leaves your machine.
- **Optional Cloud Services**: When using Voxtral or LLM correction, audio or transcribed text is sent to API providers.
- **No Tracking**: This app does not collect, store, or transmit any user data to the developers.
- **Checkpoints**: Processing checkpoints are stored locally in `chat_checkpoints/` and contain transcriptions. This directory is in `.gitignore` and is never uploaded.

### API Provider Privacy

When using cloud services, be aware:
- **Mistral AI (Voxtral)**: Audio files sent for transcription. See [Mistral Privacy Policy](https://mistral.ai/privacy-policy)
- **OpenAI**: Transcribed text sent for correction. See [OpenAI Privacy Policy](https://openai.com/privacy)
- **Anthropic (Claude)**: Transcribed text sent for correction. See [Anthropic Privacy Policy](https://www.anthropic.com/privacy)
- **Ollama**: 100% local, no data leaves your computer

### Security Best Practices

1. **Never commit `.env` file** - Use `.env.example` as template
2. **Rotate API keys** if accidentally exposed
3. **Review checkpoints** before sharing - they contain message content
4. **Use local options** (faster-whisper + Ollama) for maximum privacy

### What Data Stays Local

✅ All voice audio files (never uploaded)
✅ Chat text files (never uploaded)
✅ Media files (never uploaded)
✅ Checkpoints with transcriptions (never uploaded)
✅ API keys in `.env` file (never uploaded)

### What Data Gets Sent (Optional Cloud Services Only)

When using **Voxtral transcription**:
- Audio files sent to Mistral AI for transcription
- Transcribed text returned and stored locally

When using **LLM correction** (Claude/OpenAI/Mistral):
- Transcribed text sent for grammar/punctuation correction
- Corrected text returned and stored locally
- Original audio never sent

When using **Ollama** (local):
- Nothing sent anywhere, 100% offline

## > FAQ

**Q: Does this work without internet?**
A: Yes! Voice transcription works completely offline. Internet is only needed for optional AI correction with Claude/OpenAI.

**Q: How long does processing take?**
A: Depends on chat size and settings. Typical chat with 50 voice messages: 5-10 minutes with base model.

**Q: Can I process group chats?**
A: Yes! The app handles both individual and group chats.

**Q: What languages are supported?**
A: Whisper supports 99+ languages. It auto-detects the language in voice messages.

**Q: Is this safe to use?**
A: Yes! All code is open source and runs locally. When using cloud AI providers, only transcribed text (not audio) is sent.

**Q: Can I use this on multiple chats?**
A: Yes! Process one at a time. Use "Process Another Chat" button when done.

**Q: Do I need a GPU?**
A: No, but it helps. Faster-whisper works on CPU, just slower.

**Q: How much does it cost?**
A: Free! Optional AI correction:
- Ollama: Free
- Claude/OpenAI: Pay per API call (~$0.001-0.01 per message)

## =� System Requirements

**Minimum:**
- Python 3.9+
- 4 GB RAM
- 2 GB free disk space

**Recommended:**
- Python 3.10+
- 8 GB RAM
- 5 GB free disk space
- GPU with CUDA support (for faster transcription)

**Supported Platforms:**
- Windows 10/11
- macOS 10.15+
- Linux (Ubuntu 20.04+, or similar)

## =� License

This project is open source and available for personal use.

## =O Credits

Built with:
- [Streamlit](https://streamlit.io) - Web framework
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - Audio transcription
- [Anthropic Claude](https://anthropic.com) - AI correction (optional)
- [OpenAI](https://openai.com) - AI correction (optional)
- [Ollama](https://ollama.ai) - Local AI (optional)

## = Issues & Feedback

Found a bug or have a suggestion? Please create an issue in the repository.

---

**Made with d for privacy-conscious WhatsApp users**

