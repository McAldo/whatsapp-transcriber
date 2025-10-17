# WhatsApp Transcriber - User Guide

> Turn your WhatsApp voice messages into text automatically - 100% private and easy to use!

---

## What is WhatsApp Transcriber?

Have you ever received a long voice message on WhatsApp and wished you could just read it instead of listening? Or wanted to search through old conversations but couldn't remember if someone said something in text or voice?

**WhatsApp Transcriber solves this problem!** It's a free tool that:

✨ **Converts all voice messages to text** - Automatically transcribes every voice message in your chat

📱 **Works with any WhatsApp chat** - Individual or group conversations, Android or iPhone

🔒 **100% Private** - Everything runs on your own computer, your messages never leave your device

🔄 **Never lose progress** - Automatic checkpoints let you resume if processing is interrupted

🎯 **Easy to use** - No coding knowledge needed, simple point-and-click interface

📄 **Creates beautiful timelines** - Get a complete readable record of your entire conversation

---

## Why Use WhatsApp Transcriber?

### Common Use Cases

**📚 Keep Important Memories**
- Family conversations with elderly relatives
- Voice messages from loved ones you want to preserve
- Children's voice messages as they grow up

**💼 Work & Business**
- Meeting notes shared via voice
- Important instructions from colleagues
- Business discussions you need to reference later

**📖 Searchable Archives**
- Find specific information from old conversations
- Search through voice messages like regular text
- Create backups of important chats

**♿ Accessibility**
- Read messages in quiet environments (library, meetings)
- For people who are hard of hearing
- Easier to review long voice messages

**🌍 Language Learning**
- See transcriptions alongside voice messages
- Study pronunciation and written form together
- Practice reading in another language

---

## Features

### Core Features

**🎤 Automatic Voice Transcription**
- Transcribes all voice messages in your chat
- Works with 99+ languages automatically
- High-quality AI-powered transcription

**💻 Runs on Your Computer**
- No upload to websites or cloud services
- Your privacy is protected
- No internet required for basic features

**✨ AI Enhancement (Optional)**
- Fix transcription errors automatically
- Add proper punctuation
- Make text more readable
- Choose from multiple AI providers (some free!)

**📅 Date Range Filtering**
- Process only specific time periods
- Great for long conversations
- One-click presets (Last 7 days, Last month, etc.)

**📄 Multiple Output Formats**
- **Markdown** - Beautiful formatted text (recommended)
- **Plain Text** - Simple and universal
- **CSV** - Open in Excel or Google Sheets
- **JSON** - For technical users

**🔄 Checkpoint & Resume (NEW!)**
- Progress automatically saved as you go
- Resume interrupted processing without losing work
- No need to re-process already completed files
- Settings restored automatically

**🖼️ Organized Media Files**
- All photos, videos, and documents organized in one place
- Files renamed with dates and times
- Easy to find specific media

**📱 Mobile Friendly**
- Use from your phone or tablet
- Access from anywhere on your home WiFi
- No need to stay at your computer

**🌍 Full Unicode Support**
- Proper handling of all languages
- Italian accents (è, à, ò, ù) display correctly
- No more corrupted characters

---

## Quick Start Guide

### Step 1: Install the App

**What You Need:**
- A Windows, Mac, or Linux computer
- At least 4 GB of free space
- No special technical skills required!

**Installation (Simple):**

1. **Download Python** (if you don't have it)
   - Go to [python.org/downloads](https://python.org/downloads)
   - Download Python 3.9 or newer
   - Run the installer (check "Add Python to PATH")

2. **Download WhatsApp Transcriber**
   - Download the application files
   - Unzip to a folder (e.g., "WhatsApp Transcriber")

3. **Open a command window**
   - **Windows:** Press Windows key, type "cmd", press Enter
   - **Mac:** Press Cmd+Space, type "terminal", press Enter

4. **Navigate to the folder**
   ```
   cd "path/to/WhatsApp Transcriber"
   ```

5. **Install required components** (one-time only)
   ```
   pip install -r requirements.txt
   ```
   _(This downloads the necessary software. Takes 2-5 minutes)_

6. **Start the app**
   ```
   streamlit run app.py
   ```
   _(A web page will open automatically!)_

**That's it!** The app is now running at `http://localhost:8501`

---

### Step 2: Export Your WhatsApp Chat

**On iPhone:**

1. Open WhatsApp
2. Open the chat you want to transcribe
3. Tap the contact/group name at the top
4. Scroll down and tap "Export Chat"
5. Choose "Attach Media"
6. Save the ZIP file (to Files app or email yourself)
7. Transfer the ZIP to your computer

**On Android:**

1. Open WhatsApp
2. Open the chat you want to transcribe
3. Tap the three dots ⋮ in the top right
4. Tap "More" → "Export chat"
5. Choose "Include Media"
6. Save the ZIP file
7. Transfer the ZIP to your computer

**💡 Tip:** For very long chats with lots of media, you might want to filter by date range later to process smaller portions.

---

### Step 3: Process Your Chat

Now the fun part! Here's how to transcribe your chat:

**1. Upload Your File**
   - Click "Browse files" or drag-and-drop your ZIP file
   - Wait a moment while it loads

**2. Preview Your Chat** (Optional)
   - See conversation name, participants, date range
   - View how many messages and voice messages
   - Filter by date if needed

**3. Configure Settings**

   **Transcription Model** (How accurate?)
   - **Recommended:** "base" - Fast and accurate for most people
   - If unclear audio: Try "small" or "medium" (slower but better)
   - Testing/fast preview: "tiny" (quickest but basic quality)

   **Output Format** (How do you want to read it?)
   - **Recommended:** Markdown (.md) - Beautiful formatting
   - Plain Text (.txt) - Simple, works anywhere
   - CSV (.csv) - Open in Excel
   - JSON (.json) - For programmers

   **AI Enhancement** (Make it better? - Optional)
   - **Recommended for beginners:** Skip this for now
   - Fixes errors, adds punctuation, improves readability
   - See "AI Enhancement" section below for details

**4. Click "Process Chat"**
   - Watch the progress bar
   - Processing time depends on:
     - Number of voice messages
     - Which model you chose
     - Your computer speed
   - Typical chat with 50 voice messages: 5-10 minutes

**5. Download Your Results**
   - **Timeline file** - Your transcribed conversation
   - **Media ZIP** - All photos, videos, and files organized

**Done!** You now have a fully transcribed, searchable record of your conversation!

---

## What If Processing Gets Interrupted?

**Don't worry - your progress is automatically saved!**

### How the Checkpoint System Works

The app saves your progress automatically after every voice message is transcribed. If something goes wrong (computer crashes, you accidentally close the app, power outage, etc.), you can easily resume where you left off.

### How to Resume Processing

1. **Restart the app**
   ```
   streamlit run app.py
   ```

2. **Upload the SAME ZIP file** you used before
   - Must be the exact same file
   - App recognizes it by chat name and file count

3. **Look for the Resume Button**
   - You'll see a yellow warning box: "⚠️ Previous processing found for this chat"
   - Shows how many files were already processed
   - Shows costs so far
   - Shows when it was last processed

4. **Three options appear:**
   - **Resume Processing** - Continue from where you stopped (recommended)
   - **Start Fresh** - Discard previous work and start over
   - **Retry Failed Files** - Only re-process files that failed

5. **Click "Resume Processing"**
   - App skips files already completed
   - Your settings (transcription engine, LLM options) are automatically restored
   - Processing continues from the next unprocessed file

### What Gets Saved in Checkpoints?

- ✅ All completed transcriptions
- ✅ Your configuration (transcription engine, LLM settings, etc.)
- ✅ Cost tracking (transcription and LLM costs)
- ✅ Which files succeeded and which failed
- ✅ Processing log with timestamps

### Where Are Checkpoints Stored?

Checkpoints are saved in a folder called `chat_checkpoints/` in your app directory. Each conversation gets its own checkpoint file.

**You can safely delete checkpoints** when you're done with a chat and don't need to resume anymore.

### Example Scenario

**Situation:** You start processing a large group chat with 200 voice messages. After completing 150 transcriptions (75% done), your computer crashes.

**What happens:**
1. You restart your computer
2. Start the app again
3. Upload the same ZIP file
4. App shows: "⚠️ Previous processing found: 150/200 files completed"
5. Click "Resume Processing"
6. App processes only the remaining 50 files
7. **You saved ~30 minutes and avoided re-processing 150 files!**

### Troubleshooting Resume

**"Resume button doesn't appear"**
- Make sure you uploaded the exact same ZIP file
- Check that `chat_checkpoints/` folder exists
- Run `python check_checkpoint.py` to see if checkpoint exists

**"Want to see checkpoint details"**
- Run: `python check_checkpoint.py`
- Shows all checkpoints, progress, costs, and status

**"Characters look corrupted (è shows as �)"**
- Run: `python repair_checkpoint_encoding.py --dry-run`
- This will check and fix encoding issues
- Remove `--dry-run` to actually fix the checkpoint

---

## Understanding AI Enhancement (Optional)

AI Enhancement makes transcriptions more readable by fixing errors and adding punctuation. **This is completely optional** - your basic transcription works great without it!

### Why Use AI Enhancement?

**Before AI Enhancement:**
```
hey how are you doing um i wanted to ask you about that thing we talked about yesterday you know the project thing
```

**After AI Enhancement:**
```
Hey, how are you doing? I wanted to ask you about that thing we talked about yesterday - you know, the project thing.
```

Much more readable!

### AI Enhancement Options

**1. Ollama (FREE & PRIVATE) - Recommended for Most Users**

**What it is:** Free AI that runs on your computer

**Pros:**
- ✅ Completely FREE forever
- ✅ 100% private (nothing leaves your computer)
- ✅ No account or API key needed
- ✅ Works offline

**Cons:**
- ⚠️ Requires ~4 GB extra disk space
- ⚠️ Slower on older computers

**How to set up:**
1. Download Ollama from [ollama.ai](https://ollama.ai)
2. Install it (like any other app)
3. Open terminal/command prompt
4. Type: `ollama pull llama3.2`
5. Wait for download (one-time, ~2 GB)
6. In WhatsApp Transcriber, enable "AI correction"
7. Select "Ollama (Local)"
8. Click "Test Connection"
9. You're ready!

**💡 This is the best option for most people!**

---

**2. Claude AI (Paid but High Quality)**

**What it is:** Professional AI service by Anthropic

**Pros:**
- ✅ Excellent quality
- ✅ Very fast
- ✅ Works on any computer

**Cons:**
- 💰 Costs money (~$0.001-0.01 per message)
- 🌐 Requires internet
- 📤 Transcriptions sent to their servers

**Cost example:** 50 voice messages ≈ $0.50

**How to set up:**
1. Sign up at [console.anthropic.com](https://console.anthropic.com)
2. Add payment method (usually includes $5 free credit)
3. Create an API key
4. In WhatsApp Transcriber app:
   - Enable "AI correction"
   - Select "Claude API"
   - Enter your API key
5. Process as normal

---

**3. ChatGPT / OpenAI (Paid)**

**What it is:** AI service by OpenAI (makers of ChatGPT)

**Pros:**
- ✅ High quality
- ✅ Fast processing
- ✅ Familiar brand

**Cons:**
- 💰 Costs money (~$0.001-0.02 per message)
- 🌐 Requires internet
- 📤 Transcriptions sent to their servers

**Cost example:** 50 voice messages ≈ $0.10-1.00 (depends on mode)

**How to set up:**
1. Sign up at [platform.openai.com](https://platform.openai.com)
2. Add payment method
3. Create an API key
4. In WhatsApp Transcriber:
   - Enable "AI correction"
   - Select "OpenAI API"
   - Enter your API key
5. Process as normal

---

**4. Mistral AI (Paid)**

**What it is:** European AI service

**Pros:**
- ✅ Good quality
- ✅ Competitive pricing
- ✅ Fast processing

**Cons:**
- 💰 Costs money (~$0.002-0.01 per message)
- 🌐 Requires internet
- 📤 Transcriptions sent to their servers

**How to set up:**
1. Sign up at [console.mistral.ai](https://console.mistral.ai)
2. Create an API key
3. In WhatsApp Transcriber:
   - Enable "AI correction"
   - Select "Mistral AI"
   - Enter your API key
5. Process as normal

---

### Which AI Should I Choose?

**For most people: Use Ollama (Free & Private)**
- Best for: Anyone who wants good quality without costs
- Setup time: 10 minutes
- Ongoing cost: $0

**For best quality: Use Claude**
- Best for: Important conversations, professional use
- Cost: ~$0.50 for typical chat
- Very high accuracy

**For moderate cost: Use OpenAI**
- Best for: Good balance of cost and quality
- Cost: ~$0.10-1.00 for typical chat
- Well-known brand

**Skip AI Enhancement if:**
- You're just testing the app
- Transcription quality is already good
- You want fastest processing
- You don't want to set up anything extra

---

## Advanced Features

### Date Range Filtering

**What it does:** Process only messages from a specific time period

**Why useful:**
- Long conversations take a long time to process
- You only need recent messages
- Focus on specific events or time periods

**How to use:**

1. After uploading your chat, look for "Date Range Filter"
2. Check "Enable date filtering"
3. **Quick presets:**
   - Click "Last 7 Days" for the past week
   - Click "Last 30 Days" for the past month
   - Click "Last 90 Days" for the past 3 months
   - Click "All Time" to process everything
4. **Or choose manually:**
   - Pick start date
   - Pick end date
5. See the count of messages update automatically
6. Continue with "Process Chat"

**Example:** You have a 2-year group chat with 10,000 messages. You only need transcriptions from this month. Enable date filtering and select "Last 30 Days" - much faster!

---

### Correction Modes (Advanced AI Users)

If you're using AI enhancement, you can choose how it works:

**Message-by-Message (Default)**
- Each voice message corrected separately
- Safe for any size conversation
- More expensive for cloud AI ($0.10 for 50 messages)
- Recommended for: Beginners, large conversations

**Full Transcript (Advanced)**
- Entire conversation corrected at once
- Much faster (70% time savings)
- Cheaper for cloud AI ($0.03 for 50 messages)
- Better context understanding
- Recommended for: Small-to-medium chats, experienced users

**When to use Full Transcript:**
- Less than 50 voice messages
- Using Claude, OpenAI, or Mistral
- Want to save money
- App shows "within limits" ✓

**When to use Message-by-Message:**
- More than 100 voice messages
- Using Ollama
- Not sure what to pick (safest choice)

---

## Tips & Best Practices

### Getting Best Transcription Quality

**✅ Do's:**
- Use "base" or "small" model for good quality
- Keep original audio files (don't compress)
- Use "Auto-detect" for language (it's smart!)
- Let it process completely (don't close browser)

**❌ Don'ts:**
- Don't use "tiny" model for important chats (quality suffers)
- Don't process during other heavy tasks (slows down)
- Don't close the app while processing (will lose progress)

### Processing Large Conversations

**If you have a huge chat (5000+ messages):**

1. **Use date filtering** - Process in smaller chunks
   - This month
   - Last month
   - Previous months separately

2. **Start with a small test**
   - Filter to just last 7 days first
   - Make sure settings are right
   - Then process larger ranges

3. **Be patient**
   - Large chats take time (30-60+ minutes)
   - Progress bar shows status
   - Let your computer focus on this task

4. **Don't worry about interruptions**
   - Progress is automatically saved
   - You can resume anytime
   - Even close the app and come back later

### Saving Money on AI (If Using Paid Options)

**1. Test without AI first**
   - Process without AI enhancement
   - See if quality is already good enough
   - Only add AI if needed

**2. Use bulk mode**
   - For chats under 50 messages
   - Saves 70% on API costs
   - Much faster too

**3. Filter to what you need**
   - Don't process entire 2-year chat
   - Pick the specific months you need

**4. Consider Ollama (Free!)**
   - One-time 10-minute setup
   - Free forever after that
   - Quality is quite good

---

## Troubleshooting

### "Can't find chat file in ZIP"

**Problem:** App can't find your WhatsApp messages

**Solution:**
- Make sure you exported from WhatsApp (not just copied files)
- Make sure you chose "Include Media" or "Attach Media"
- Try exporting again
- Check the ZIP file contains a `_chat.txt` file

---

### Processing is Very Slow

**Problem:** Transcription taking forever

**Possible causes & solutions:**

**1. Using CPU instead of GPU**
   - Do you have an NVIDIA graphics card?
   - Check if app says "CPU" in the status
   - See "Enable GPU Acceleration" section in README
   - **Quick fix:** Use a smaller model (tiny or base)

**2. Large model on slow computer**
   - Using "medium" or "large" model?
   - **Solution:** Switch to "base" (still good quality!)

**3. Many voice messages**
   - 100+ voice messages takes time (20-40 minutes)
   - This is normal!
   - Use date filtering to process fewer messages

---

### "Characters look weird (è shows as � or strange codes)"

**Problem:** Italian or other language accents not displaying correctly

**Solution:**
1. **Open timeline in proper text editor:**
   - Use Notepad++ (Windows) - Set encoding to "UTF-8-BOM"
   - Use TextEdit (Mac) - Should auto-detect
   - Use VS Code - Set encoding to "UTF-8 with BOM"

2. **If problem persists in checkpoint:**
   ```bash
   # Check for encoding issues
   python repair_checkpoint_encoding.py --dry-run

   # Fix the checkpoint
   python repair_checkpoint_encoding.py
   ```

3. **Regenerate timeline:**
   - Resume processing (will use fixed checkpoint)
   - New timeline will have correct characters

---

### "I accidentally closed the app during processing!"

**Problem:** Processing was interrupted

**Solution:**
1. **Don't panic** - Your progress is saved!
2. **Restart the app:**
   ```
   streamlit run app.py
   ```
3. **Upload the same ZIP file**
4. **Click "Resume Processing"** when the button appears
5. Processing continues from where it stopped

**Your completed transcriptions are safe!**

---

### "Resume button doesn't appear but I know I was processing this chat"

**Problem:** Checkpoint exists but not detected

**Solutions:**

**1. Check if checkpoint exists:**
```bash
python check_checkpoint.py
```
This shows all saved checkpoints and their status

**2. Make sure you uploaded the exact same ZIP file**
- Same file, not a new export
- File name can be different, but contents must match

**3. Check chat name:**
- If WhatsApp added (2), (3) to filename, it might not match
- The diagnostic tool above will show the chat name in checkpoint

**4. Manual recovery (advanced):**
- Check `chat_checkpoints/` folder
- Your transcriptions are saved in JSON format
- Can be manually extracted if needed

---

### App Won't Start / "Error installing"

**Problem:** Can't run the app or installation failed

**Solutions:**

**1. Python not installed correctly**
   ```
   python --version
   ```
   Should show Python 3.9 or newer
   If not: Reinstall Python, check "Add to PATH"

**2. Installation failed**
   Try:
   ```
   pip install --upgrade pip
   pip install -r requirements.txt --upgrade
   ```

**3. Windows: Microsoft Visual C++ required**
   - Download from [Microsoft](https://aka.ms/vs/17/release/vc_redist.x64.exe)
   - Install and restart computer

---

### "Out of Memory" Error

**Problem:** Computer runs out of memory during processing

**Solutions:**

**1. Use smaller model**
   - Switch from "large" → "medium" → "small" → "base"
   - "Base" uses only ~2 GB memory

**2. Close other programs**
   - Close browser tabs
   - Close video games
   - Close other memory-heavy apps

**3. Process in smaller chunks**
   - Use date filtering
   - Process one month at a time

---

### AI Enhancement Not Working

**Problem:** Ollama connection failed or API key rejected

**Solutions for Ollama:**
```bash
# Check if Ollama is running
ollama list

# If not running, start Ollama app

# Make sure model is downloaded
ollama pull llama3.2

# Test it works
ollama run llama3.2 "Hello"
```

**Solutions for Cloud AI (Claude/OpenAI/Mistral):**
- Double-check API key (no extra spaces)
- Check you added payment method to account
- Verify you have credit/balance
- Try creating a new API key

---

## FAQ

**Q: Is my data safe? Where do my messages go?**

A: By default, **everything stays on your computer**. Voice messages are transcribed locally, nothing is uploaded. Only if you enable AI enhancement with Claude/OpenAI/Mistral, the transcribed TEXT (not audio) is sent to their servers for correction. Using Ollama keeps everything 100% local.

---

**Q: How long does it take to process?**

A: Depends on:
- **10 voice messages:** ~2-5 minutes
- **50 voice messages:** ~10-15 minutes
- **200 voice messages:** ~45-60 minutes

With date filtering, you can process faster by doing smaller chunks.

---

**Q: Can I process group chats?**

A: Yes! Works perfectly with both individual and group conversations.

---

**Q: What languages does it support?**

A: Over 99 languages! Including English, Spanish, French, German, Italian, Portuguese, Arabic, Hindi, Chinese, Japanese, Korean, and many more. The app auto-detects the language.

---

**Q: Do I need an internet connection?**

A: For basic transcription with faster-whisper: **No, works offline!**

For AI enhancement:
- Ollama: No (works offline)
- Claude/OpenAI/Mistral: Yes (internet required)
- Voxtral transcription: Yes (internet required)

---

**Q: Can I use this on multiple chats?**

A: Yes! Process as many chats as you want, one at a time. After each chat, click "Process Another Chat" button.

---

**Q: How much does it cost?**

A: **The app itself is 100% free!**

Optional costs:
- Basic transcription: **Free**
- Ollama AI enhancement: **Free**
- Claude AI enhancement: ~$0.01 per message (~$0.50 typical chat)
- OpenAI AI enhancement: ~$0.002-0.02 per message (~$0.10-1.00 typical chat)
- Mistral AI enhancement: ~$0.002-0.01 per message
- Voxtral transcription: ~$0.001 per minute of audio

---

**Q: Do I need a powerful computer?**

A: **No!** The app works on any computer with:
- 4 GB RAM (8 GB recommended)
- 2-5 GB free disk space
- Any relatively modern CPU

Having a gaming GPU (NVIDIA) makes it 5-10x faster, but not required.

---

**Q: Can I search the transcriptions?**

A: Yes! Once you download the timeline file, you can:
- Open in any text editor
- Use Ctrl+F (Cmd+F on Mac) to search
- Open CSV in Excel and use filter/search features

---

**Q: What if transcription quality is poor?**

A: Try these in order:
1. Use a larger model (base → small → medium)
2. Enable AI enhancement (Ollama is free!)
3. Check if original audio is clear (background noise affects quality)

---

**Q: Can I use this on my phone?**

A: You need a computer to run the app, but you can **access it from your phone** once it's running:
1. Start app on computer
2. Make sure phone is on same WiFi
3. Open phone browser
4. Go to `http://{your-computer-ip}:8501`
5. Upload and process from your phone!

---

**Q: Is this legal?**

A: Yes! You're processing your own WhatsApp exports on your own computer. This is perfectly legal. The app is open-source and free to use.

---

**Q: Can I export just the transcriptions without the full timeline?**

A: The timeline includes transcriptions. To get just the text:
- Choose CSV or JSON format
- Open in Excel/text editor
- Filter/extract transcription column

---

## Getting Help

**Something not working?**

1. **Check this User Guide** - Most answers are here
2. **Read the error message** - Often tells you what to fix
3. **Try the README file** - Has detailed troubleshooting
4. **Ask for help:**
   - GitHub Issues (for bug reports)
   - GitHub Discussions (for questions)

---

## What's Next?

Once you're comfortable with the basics:

1. **Try AI Enhancement** - Make transcriptions more readable
2. **Experiment with Models** - Find the balance of speed vs quality for you
3. **Use Date Filtering** - Process specific time periods efficiently
4. **Try Different Output Formats** - Find what works best for your needs

---

**Enjoy your transcribed conversations! 🎉**

*Made with ❤️ for people who love to read (instead of listen)*
