# Claude Code Prompt: Prepare and Upload WhatsApp Transcriber to GitHub

## Context
WhatsApp Transcriber project is ready to be shared on GitHub. Need to ensure all sensitive data is protected and repository is properly configured.

## Pre-Upload Tasks

### Task 1: Create/Update .gitignore

Create `.gitignore` file in project root with the following content:

```gitignore
# Environment variables (CRITICAL - NEVER upload!)
.env
.env.local
.env.*.local

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/
env.bak/
venv.bak/

# User data (private - contains user messages)
chat_checkpoints/
*.zip
*.opus
*.mp3
*.m4a
*.aac
*.wav

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
desktop.ini

# Logs
*.log
logs/

# Temporary files
tmp/
temp/
*.tmp
*.temp

# Testing
.pytest_cache/
.coverage
htmlcov/

# Streamlit
.streamlit/secrets.toml
```

### Task 2: Create .env.example

Create `.env.example` file as a template (safe to upload):

```bash
# WhatsApp Transcriber - Environment Variables Template
# Copy this file to .env and fill in your actual API keys

# Mistral AI API Key (for Voxtral transcription)
# Get your key at: https://console.mistral.ai/
MISTRAL_API_KEY=your_mistral_api_key_here

# OpenAI API Key (for LLM correction - optional)
# Get your key at: https://platform.openai.com/
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic Claude API Key (for LLM correction - optional)
# Get your key at: https://console.anthropic.com/
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Ollama URL (for local LLM - optional)
# Default: http://localhost:11434
OLLAMA_URL=http://localhost:11434
```

### Task 3: Create LICENSE File

Create `LICENSE` file with MIT License:

```
MIT License

Copyright (c) 2025 [Project Author]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Task 4: Verify No Sensitive Data in Code

**CRITICAL SECURITY CHECK:**

1. Search entire project for API keys:
   ```bash
   # Search for potential API keys in code
   grep -r "sk-ant-" . --exclude-dir=venv --exclude-dir=.git
   grep -r "sk-proj-" . --exclude-dir=venv --exclude-dir=.git
   grep -r "mistral_api_key.*=" . --exclude-dir=venv --exclude-dir=.git
   ```

2. Verify `.env` is NOT in repository:
   ```bash
   git ls-files | grep "\.env$"
   # Should return nothing
   ```

3. Check for hardcoded keys in Python files:
   - Open each `.py` file
   - Look for patterns like: `api_key = "sk-..."`
   - Replace with: `api_key = os.environ.get('MISTRAL_API_KEY')`

4. Verify no user data:
   - Check `chat_checkpoints/` is gitignored
   - Verify no `.opus`, `.zip` files in repository

### Task 5: Add Privacy Notice to README

Add this section to `README.md` under a "Privacy & Security" heading:

```markdown
## 🔒 Privacy & Security

### Data Processing

- **Local by Default**: Voice transcription happens on your computer using faster-whisper. No data leaves your machine.
- **Optional Cloud Services**: When using Voxtral or LLM correction, only transcribed text (not audio) is sent to API providers.
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
```

### Task 6: Update README with Installation from GitHub

Add installation instructions to README:

```markdown
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
```

## Git Commands to Execute

### Initialize Repository (if not already done)

```bash
# Navigate to project directory
cd C:\Android_Projects\whatsapp-transcriber

# Initialize git repository
git init

# Add all files (respecting .gitignore)
git add .

# Create initial commit
git commit -m "Initial commit: WhatsApp Transcriber with checkpoint system

Features:
- Local voice transcription with faster-whisper (100% offline)
- Cloud transcription with Voxtral Mini (Mistral AI) for superior Italian quality
- Optional LLM correction (Claude, OpenAI, Mistral, Ollama)
- Checkpoint & resume system - automatic progress tracking, never lose work
- Two correction modes: Message-by-Message (reliable) and Bulk (faster, cheaper)
- Token estimation with automatic limit checking
- Date range filtering with quick presets
- Multiple output formats (Markdown, TXT, CSV, JSON)
- UTF-8-BOM encoding for proper Italian/special character support
- Detailed processing logs and diagnostic tools
- Mobile-friendly web interface with LAN access"
```

### Create GitHub Repository

**Option A: Via GitHub Website (Recommended for beginners)**

1. Go to [github.com](https://github.com) and sign in
2. Click the "+" icon (top right) → "New repository"
3. Repository name: `whatsapp-transcriber`
4. Description: "Privacy-focused WhatsApp chat processor with voice transcription and AI enhancement"
5. Choose: **Public** (to share) or **Private** (keep private)
6. **DO NOT** initialize with README (you already have one)
7. Click "Create repository"
8. Follow the instructions shown for "push an existing repository"

**Option B: Via GitHub CLI (if installed)**

```bash
# Create repository on GitHub
gh repo create whatsapp-transcriber --public --source=. --remote=origin --push

# Or for private repository
gh repo create whatsapp-transcriber --private --source=. --remote=origin --push
```

### Connect Local Repository to GitHub

After creating repository on GitHub, connect your local repository:

```bash
# Add GitHub as remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/whatsapp-transcriber.git

# Verify remote was added
git remote -v

# Push to GitHub
git branch -M main
git push -u origin main
```

### Verify Upload Success

After pushing, check:

1. **Go to your GitHub repository URL**
2. **Verify files are present**:
   - ✅ All code files (`.py`)
   - ✅ Documentation files:
     - `README.md` (user-facing guide)
     - `DEVELOPER_README.md` (developer guide)
     - `TECHNICAL_OVERVIEW.md` (architecture details)
     - `USER_GUIDE.md` (comprehensive user manual)
   - ✅ `requirements.txt`
   - ✅ `.gitignore`
   - ✅ `LICENSE`
   - ✅ `.env.example`
   - ✅ Diagnostic tools (if present):
     - `check_checkpoint.py`
     - `repair_checkpoint_encoding.py`
   - ❌ NO `.env` file
   - ❌ NO `chat_checkpoints/` directory
   - ❌ NO `venv/` directory
   - ❌ NO API keys visible anywhere

3. **Test clone on another machine** (if possible):
   ```bash
   git clone https://github.com/YOUR_USERNAME/whatsapp-transcriber.git
   cd whatsapp-transcriber
   pip install -r requirements.txt
   streamlit run app.py
   ```

## Post-Upload Tasks

### Task 1: Add Repository Topics

On GitHub repository page:
1. Click "⚙️ Settings"
2. In "About" section, click "⚙️" (gear icon)
3. Add topics:
   - `whatsapp`
   - `transcription`
   - `voice-to-text`
   - `streamlit`
   - `whisper`
   - `python`
   - `mistral-ai`
   - `privacy`
   - `checkpoint-system`
   - `resume-processing`
   - `llm`
   - `ai-enhancement`

### Task 2: Create Releases (Optional)

Create a release for version tracking:
1. Go to repository → "Releases"
2. Click "Create a new release"
3. Tag version: `v1.0.0`
4. Release title: `v1.0.0 - Initial Public Release`
5. Description: List key features
6. Click "Publish release"

### Task 3: Add Repository Description

On repository main page, click "⚙️" next to About and add:
- **Description**: "Privacy-focused WhatsApp voice transcriber with automatic checkpoint/resume system. Process voice messages locally with Whisper or via Mistral AI's Voxtral API. Never lose progress - resumes automatically after interruptions. Optional LLM enhancement with Claude, OpenAI, Mistral, or local Ollama. Full UTF-8 support for all languages."
- **Website**: (if you have one)
- **Topics**: (added above)

### Task 4: Enable GitHub Issues (Optional)

For community bug reports and feature requests:
1. Go to Settings
2. Scroll to "Features"
3. Check "Issues"

## Security Checklist

Before making repository public, verify:

- [ ] `.env` file is in `.gitignore`
- [ ] No API keys in any code files
- [ ] No API keys in commit history
- [ ] `chat_checkpoints/` in `.gitignore`
- [ ] No user data (messages, audio files) in repository
- [ ] `.env.example` has placeholder values only
- [ ] Virtual environment (`venv/`) in `.gitignore`
- [ ] All sensitive directories in `.gitignore`
- [ ] Privacy notice added to README
- [ ] License file present
- [ ] Installation instructions clear

## If API Key Was Accidentally Committed

**CRITICAL**: If you accidentally committed `.env` or API keys:

1. **Rotate API keys immediately** (generate new ones on provider websites)
2. **Remove from git history**:
   ```bash
   # Install BFG Repo Cleaner or use git filter-branch
   git filter-branch --force --index-filter \
   "git rm --cached --ignore-unmatch .env" \
   --prune-empty --tag-name-filter cat -- --all
   
   # Force push (this rewrites history)
   git push origin --force --all
   ```
3. **Verify removal** by cloning fresh copy
4. **Never reuse exposed keys**

## Success Criteria

✅ Repository created on GitHub
✅ All files uploaded successfully
✅ No sensitive data in repository
✅ `.env.example` present as template
✅ `.gitignore` properly configured
✅ Documentation complete and accurate
✅ License file included
✅ Privacy notice added
✅ Installation instructions clear
✅ Repository description and topics added
✅ Can clone and run on fresh machine

## Notes

- **First-time Git users**: GitHub Desktop app is easier than command line
- **SSH vs HTTPS**: HTTPS is simpler for beginners (no SSH key setup needed)
- **Private vs Public**: Start private, make public when ready
- **Collaborators**: Can be added later in Settings → Collaborators
- **Fork protection**: Enable "Require pull request reviews" in Settings for collaboration

---

**Ready to share your project with the world! 🚀**
