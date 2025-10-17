I'm working in a Python virtual environment. The virtual environment is already created and activated.

Please create the WhatsApp transcriber application as described below.

Build a Python Streamlit application that processes WhatsApp chat exports and creates a comprehensive timeline with transcriptions. This app will run locally on a computer and be accessible from mobile devices on the same network.

REQUIREMENTS:

1. FILE HANDLING:
   - Accept ZIP file upload (WhatsApp export)
   - Extract all contents to temporary directory
   - Parse the main chat text file (typically named "_chat.txt" or "WhatsApp Chat with [Name].txt")
   - Identify all .opus audio files and other media files
   - Handle large files efficiently (show file size limits in UI)

2. TRANSCRIPTION:
   - Use faster-whisper to transcribe all .opus audio files locally (NO API KEY NEEDED - runs locally)
   - Model size should be configurable in UI (base/small/medium/large)
   - Extract timestamp from audio filename format: PTT-YYYYMMDD-WA####.opus
   - Note: Exact time not available in filename, use date only or check file metadata if available
   - Show estimated time remaining during transcription

3. LLM CORRECTION (OPTIONAL):
   - Add UI toggle to enable/disable transcription correction
   - Support THREE options (dropdown selection):
     * Claude API
     * OpenAI API
     * Ollama (local model - RECOMMENDED for privacy)
   
   - HYBRID API KEY HANDLING (for Claude/OpenAI):
     * First check for environment variables: ANTHROPIC_API_KEY or OPENAI_API_KEY
     * If not found, show password input field in UI for user to enter key
     * Display message indicating if key loaded from environment or needs manual input
   
   - OLLAMA CONFIGURATION (RECOMMENDED - FREE & PRIVATE):
     * Ollama server URL input (default: http://localhost:11434)
     * Model name dropdown/input with popular suggestions:
       - llama3.2 (recommended - lightweight)
       - llama3.1
       - mistral
       - gemma2
     * "Test Connection" button to verify Ollama is running
     * Display clear instructions if Ollama not detected
     * No API key needed
   
   - Correct transcription errors, improve punctuation, fix obvious mistakes
   - Keep corrections minimal and faithful to original speech
   - Batch corrections for efficiency

4. TIMELINE GENERATION:
   - Combine all messages chronologically:
     * Text messages from chat export (with original timestamps)
     * Transcribed voice messages (with date from filename)
     * References to media files sent (images, docs, videos)
   - Parse WhatsApp timestamp format from chat file (handle multiple formats)
   - Output in multiple formats (user selectable):
     * Markdown (.md) - default, most readable
     * Plain text (.txt)
     * CSV (.csv)
     * JSON (.json)
   - Include message sender identification

5. MEDIA ORGANIZATION:
   - Copy all media files to output folder
   - Rename files: {YYYYMMDD_HHMMSS}_{original_filename} (extract timestamp from WhatsApp metadata if available)
   - Keep all files in single folder (no subfolders)
   - Maintain file type extensions
   - Create index of media files with references to timeline

6. STREAMLIT UI (MOBILE-FRIENDLY):
   - Configure for mobile access:
     * Responsive layout that works on phones
     * Collapsible sidebar for mobile
     * Full-width buttons on small screens
     * Touch-friendly controls
   
   - Main interface with sections:
     
     **Upload Section:**
     * File uploader for ZIP (with drag-drop support)
     * Display uploaded file info (name, size)
     
     **Configuration Section (Collapsible):**
     * Whisper model selector (dropdown: base/small/medium/large)
       - Show estimated speed/quality tradeoff
       - Recommend "base" for quick processing, "small" for balance
     * Output format selector (dropdown: markdown/txt/csv/json)
     * LLM correction toggle (checkbox)
     * When LLM enabled: 
       - Provider dropdown (Claude API / OpenAI API / Ollama Local)
       - Highlight Ollama as "FREE & PRIVATE" option
       - For Claude/OpenAI:
         * Status: "✓ API key loaded from environment" or password input
       - For Ollama:
         * Server URL (default: http://localhost:11434)
         * Model selector with descriptions
         * "Test Connection" button with status indicator
         * Help text: "Install Ollama from ollama.ai"
     
     **Processing Section:**
     * Large "Process Chat" button
     * Real-time progress with:
       - Overall progress bar (0-100%)
       - Current step indicator with icon
       - Detailed status text:
         * "📦 Extracting files... (X files found)"
         * "📝 Parsing Y text messages..."
         * "🎤 Transcribing audio X of Y: [filename] (Z% complete)"
         * "✨ Applying LLM corrections... (X of Y messages)"
         * "📁 Organizing Z media files..."
       - Estimated time remaining
       - Cancel button to abort processing
     
     **Results Section:**
     * Success message with summary
     * Statistics card showing:
       - Total messages processed
       - Text messages: X
       - Voice messages transcribed: Y
       - Media files organized: Z
       - Processing time: MM:SS
       - LLM provider used (if enabled)
     * Two download buttons:
       - "📄 Download Timeline" (chosen format)
       - "📦 Download Media Files" (ZIP)
     * Preview of timeline (first 50 lines)
     * "Process Another Chat" button to reset
     
     **Network Access Instructions:**
     * Show local network address in sidebar
     * Example: "Access from phone: http://192.168.1.100:8501"
     * Instructions for finding computer's local IP
     * Note about same WiFi network requirement

7. OUTPUT:
   - Timeline file in chosen format containing:
     * Header with processing metadata (date, settings used)
     * All text messages with timestamps and sender
     * All transcribed voice messages with date/time and sender
     * Media file references with timestamps
     * Clear visual separation between messages
     * For markdown: use formatting for readability
   
   - Media folder (as ZIP) with:
     * All renamed media files
     * index.txt listing all files with original names and timestamps
   
   - Processing log file with:
     * Detailed processing information
     * Any errors or warnings encountered
     * Model versions used
     * Processing duration for each step

TECHNICAL REQUIREMENTS:

- Mobile-optimized Streamlit configuration:
```python
  st.set_page_config(
      page_title="WhatsApp Transcriber",
      page_icon="💬",
      layout="centered",
      initial_sidebar_state="collapsed"
  )

Add custom CSS for mobile responsiveness:

Full-width buttons on mobile
Larger touch targets
Readable font sizes
Proper spacing for touch interfaces


Use faster-whisper library for local transcription

Implement proper device detection (CPU/GPU)
Add memory management for large files


Handle LLM providers:

Claude API via anthropic library
OpenAI API via openai library
Ollama via requests (simple HTTP API calls)


API key handling with hybrid approach:

Check environment variables first
Fall back to Streamlit password input
Store in session state only
Never write keys to disk


Ollama integration:

Test connection before processing
Provide clear setup instructions
Handle connection errors gracefully
Use streaming for better UX if possible


Error handling:

Corrupted ZIP files
Missing chat text file
Invalid audio files
API/connection errors
Out of memory scenarios
User-friendly error messages


Performance optimization:

Process audio files in batches
Show progress for long operations
Clean up temporary files
Efficient memory usage for large chats


Logging:

Detailed logs for debugging
Processing statistics
Error tracking



DEPENDENCIES (requirements.txt):

streamlit>=1.28.0
faster-whisper>=0.10.0
anthropic>=0.7.0
openai>=1.3.0
requests>=2.31.0
python-dotenv>=1.0.0
pandas>=2.0.0

DELIVERABLES:

app.py - Main Streamlit application with:

Clean, well-commented code
Mobile-responsive UI
All features implemented
Error handling throughout


requirements.txt - All dependencies with version pins
README.md with:

Quick start guide
Installation instructions (pip install -r requirements.txt)
How to run locally (streamlit run app.py)
How to access from phone on local network:

Find computer IP address (Windows/Mac/Linux commands)
Connect phone to same WiFi
Open browser to http://[YOUR-IP]:8501


Optional: Environment variables setup (.env file)
Ollama setup instructions:

Download from ollama.ai
Install on your system
Pull recommended model: ollama pull llama3.2
Start Ollama (usually automatic)


Troubleshooting section:

Ollama connection issues
Phone can't connect (firewall, network)
Out of memory errors
Audio transcription problems


FAQ section


.env.example - Template for environment variables:

   # Optional: Set API keys here instead of UI
   ANTHROPIC_API_KEY=your_key_here
   OPENAI_API_KEY=your_key_here

utils/ folder (if needed) with helper functions:

whatsapp_parser.py (parse chat format)
audio_processor.py (transcription logic)
llm_corrector.py (correction logic)
file_organizer.py (media handling)



QUALITY STANDARDS:

Production-ready code with proper error handling
User-friendly for non-technical users
Clear progress feedback at every step
Works smoothly on mobile browsers
Efficient resource usage
Clean, maintainable code structure
Comprehensive documentation

Please create a complete, polished solution that works great for local use and can be easily accessed from a mobile device on the same network.



Additional notes:
- The virtual environment is located at ./venv/
- requirements.txt should list all dependencies
- app.py should be the main Streamlit application
- Create any additional helper files as needed in a utils/ folder