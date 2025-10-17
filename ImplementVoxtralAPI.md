

## Prompt for Claude Code - Add Voxtral Mini API:

```
Add Voxtral Mini (via Mistral API) as an additional transcription engine option to the WhatsApp transcriber.

IMPLEMENTATION:

1. ADD TRANSCRIPTION ENGINE SELECTOR:
   
   In the Configuration Section, add a new dropdown:
   - Label: "Transcription Engine"
   - Options:
     * "Faster-Whisper (Local)" - default
     * "Voxtral Mini (Mistral API)" - new option
   
   When "Voxtral Mini (Mistral API)" is selected:
   - Show info message: "💡 Voxtral offers better quality, especially for Italian. Cost: ~$0.001/minute"
   - Show API key input (use hybrid approach like LLM correction)

2. MISTRAL API KEY HANDLING:
   
   - First check environment variable: MISTRAL_API_KEY
   - If found: Display "✓ API key loaded from environment"
   - If not found: Show password input field for manual entry
   - Store in session state only
   - Add to .env.example file

3. VOXTRAL TRANSCRIPTION IMPLEMENTATION:
   
   Install dependency:
   ```
   mistralai>=1.0.0
   ```
   
   Code structure:
   ```python
   from mistralai import Mistral
   import os
   
   # Initialize client
   mistral_api_key = os.getenv('MISTRAL_API_KEY') or user_input_key
   client = Mistral(api_key=mistral_api_key)
   
   # For each audio file:
   def transcribe_with_voxtral(audio_path, language=None):
       try:
           # Voxtral supports .opus directly, but may need conversion for some formats
           # Check if file format is supported, convert if needed
           
           with open(audio_path, "rb") as audio_file:
               # Upload the file
               uploaded = client.files.upload(
                   file={"content": audio_file, "file_name": os.path.basename(audio_path)},
                   purpose="audio"
               )
               
               # Get signed URL
               signed_url = client.files.get_signed_url(file_id=uploaded.id)
               
               # Transcribe
               transcription = client.audio.transcriptions.complete(
                   model="voxtral-mini-latest",
                   file_url=signed_url.url,
                   language=language if language and language != "Auto-detect" else None
               )
               
               return transcription.text
               
       except Exception as e:
           # Log error and optionally fall back to Whisper
           st.error(f"Voxtral transcription failed: {str(e)}")
           return None
   ```

4. UPDATE PROGRESS INDICATORS:
   
   When using Voxtral, show:
   - "🎤 Transcribing with Voxtral Mini API: audio X of Y"
   - If auto-detect: "🎤 Transcribing (Voxtral Mini): [filename] - language detected"
   - Show estimated cost: "Estimated cost: $X.XX (based on Y minutes of audio)"

5. AUDIO FORMAT HANDLING:
   
   Voxtral API supports: .mp3, .wav, .flac, .ogg, .opus
   
   WhatsApp uses .opus, which is supported directly.
   
   Add validation:
   - Check if audio file format is supported
   - If not, show warning and skip or convert
   - Handle conversion errors gracefully

6. COST TRACKING:
   
   Add cost calculator in UI:
   - Calculate total audio duration before processing
   - Show estimated cost: duration_minutes × $0.001
   - Display running total during processing
   - Show final cost in summary: "Total Voxtral API cost: $X.XX"

7. ERROR HANDLING:
   
   Handle common errors:
   - Invalid API key → Clear error message, prompt for correct key
   - Rate limits → Show friendly message, suggest trying again
   - File upload failures → Retry logic (max 3 attempts)
   - Unsupported audio format → Skip file with warning or convert
   - Network errors → Clear message, suggest checking connection
   
   For each failed transcription:
   - Log the error
   - Optionally offer to fall back to Whisper for that file
   - Continue with remaining files

8. LANGUAGE HANDLING:
   
   Voxtral supports the same languages we already have in the dropdown:
   - Auto-detect (default, recommended)
   - Italian, English, Spanish, French, German, Portuguese, etc.
   
   When language is selected:
   - Pass language code to API (it, en, es, fr, de, pt)
   - When auto-detect: don't pass language parameter
   
   Language codes mapping:
   ```python
   language_codes = {
       "Italian": "it",
       "English": "en",
       "Spanish": "es",
       "French": "fr",
       "German": "de",
       "Portuguese": "pt",
       "Dutch": "nl",
       "Russian": "ru",
       "Chinese": "zh",
       "Japanese": "ja"
   }
   ```

9. UPDATE PROCESSING SUMMARY:
   
   Add to summary output:
   - Engine used: "Voxtral Mini (Mistral API)"
   - Total audio duration: "X minutes Y seconds"
   - API cost: "$X.XX"
   - Success rate: "X/Y files transcribed successfully"

10. UPDATE README:
    
    Add section "Transcription Engines":
    
    ```markdown
    ## Transcription Engines
    
    ### Faster-Whisper (Default)
    - Runs locally on your computer
    - Free, no API costs
    - Good quality
    - Works offline
    - CPU or GPU supported
    - Recommended for: General use, offline transcription
    
    ### Voxtral Mini (Mistral API)
    - Uses Mistral's cloud API
    - $0.001 per minute of audio (~$0.06 per hour)
    - Better quality than Whisper, especially for Italian
    - Requires internet connection
    - Requires API key (get free credits at mistral.ai)
    - Recommended for: High-quality Italian transcription
    
    #### Getting Mistral API Key:
    1. Sign up at https://console.mistral.ai/
    2. Navigate to API Keys
    3. Create new key
    4. Copy and paste into app, or add to .env file:
       ```
       MISTRAL_API_KEY=your_key_here
       ```
    
    #### Cost Examples:
    - 10 minute conversation: ~$0.01
    - 1 hour conversation: ~$0.06
    - 10 hours of audio: ~$0.60
    ```

11. UI COMPARISON HELPER:
    
    Add an expandable info box in Configuration:
    
    ```
    ℹ️ Which engine should I choose?
    
    | Feature | Faster-Whisper | Voxtral Mini API |
    |---------|----------------|------------------|
    | Quality | Good | Better |
    | Italian | Good | Excellent |
    | Speed | Fast | Fast |
    | Cost | Free | ~$0.001/min |
    | Privacy | 100% local | Sends to API |
    | Internet | Not required | Required |
    | Setup | Included | Need API key |
    ```

12. TESTING & VALIDATION:
    
    Before processing:
    - Test Mistral API connection with a small test request
    - Validate API key format
    - Show clear error if API key is invalid
    - Allow user to correct and retry

IMPORTANT NOTES:

- Keep Faster-Whisper as default (most users won't have API key)
- Make Voxtral clearly marked as "optional but better for Italian"
- Handle all errors gracefully - never crash the app
- Always show cost estimates upfront
- Make it easy to switch between engines
- Preserve all existing functionality

DEPENDENCIES:
Add to requirements.txt:
```
mistralai>=1.0.0
```

FILES TO UPDATE:
- app.py (main implementation)
- requirements.txt (add mistralai)
- README.md (documentation)
- .env.example (add MISTRAL_API_KEY)

Please implement this feature maintaining code quality, error handling, and user experience.
```
