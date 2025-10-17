
## Prompt for Claude Code - Improvements to Date Filtering and LLM Correction:

```
Please make the following improvements to the WhatsApp transcriber app:

PART 1: FIX DATE RANGE QUICK FILTERS

1. ADD "1 DAY" QUICK FILTER:
   - Add "Last 1 Day" button to the existing quick filters
   - Order: [Last 1 Day] [Last 7 Days] [Last 30 Days] [Last 3 Months] [Last Year] [Custom Range]

2. FIX LIVE COUNT UPDATE:
   - Currently, the quick filter buttons don't update the "Will process: X messages, Y audio, Z media" count
   - Make the count update immediately when:
     * Quick filter button is clicked
     * Start date is changed
     * End date is changed
     * "Process entire conversation" checkbox is toggled
   
   Implementation:
   - Use Streamlit's reactive updates (st.session_state or callbacks)
   - Recalculate filtered counts whenever date range changes
   - Display: "Will process: X text messages, Y audio files, Z media files"

PART 2: ADD BULK TRANSCRIPT CORRECTION MODE

Currently: LLM correction sends each message individually (slow, expensive)
New: Add option to send entire transcript at once (faster, cheaper)

1. ADD CORRECTION MODE SELECTOR:
   
   In the LLM Correction section, add radio buttons:
   - "Message-by-Message" (current behavior, default)
     * Description: "Corrects each transcription individually. Slower but works with any model."
   - "Full Transcript" (new option)
     * Description: "Sends entire transcript at once. Faster and cheaper, but requires large context window."

2. FULL TRANSCRIPT CORRECTION IMPLEMENTATION:

   When "Full Transcript" mode is selected:
   
   A. Build complete transcript with labels:
   ```
   [Text Message - 2025-01-15 10:23] John: Hey, how are you?
   [Text Message - 2025-01-15 10:24] Mary: I'm good, thanks!
   [Voice Message - 2025-01-15 10:25] John: [TRANSCRIPTION] Can we meet tomorrow at the cafe?
   [Voice Message - 2025-01-15 10:26] Mary: [TRANSCRIPTION] Yes, that sounds great. See you at two PM.
   [Text Message - 2025-01-15 10:27] John: Perfect!
   ```
   
   B. Token estimation:
   - Count approximate tokens (rough estimate: 1 token ≈ 4 characters)
   - Display BEFORE processing: "⚠️ Estimated tokens: ~X,XXX tokens"
   - Show warning if exceeds common limits:
     * If > 120k tokens: "⚠️ Warning: May exceed context window for some models"
     * If > 200k tokens: "❌ Error: Transcript too large. Use Message-by-Message mode."
   
   C. Send to LLM with correction prompt:
   ```
   Please review and correct any transcription errors in the following conversation transcript. 
   
   Rules:
   - Fix obvious speech-to-text errors (wrong words, missing punctuation, etc.)
   - Keep corrections minimal - only fix clear mistakes
   - Preserve the original meaning and speaker intent
   - Maintain the exact format: [Type - Date] Speaker: Content
   - Do NOT translate, summarize, or change the conversation structure
   
   Transcript:
   [paste full transcript here]
   
   Please return the corrected transcript in the same format.
   ```
   
   D. Parse corrected response:
   - Extract corrected messages from LLM response
   - Match back to original messages by timestamp/label
   - Update only the content, preserve metadata

3. UPDATE MODEL SELECTION FOR BULK MODE:

   When "Full Transcript" mode is selected:
   
   A. For OpenAI:
   - Change from current model to: "gpt-4o" or "gpt-4-turbo"
   - Show model info: "Using gpt-4o (128k context)"
   - Display cost estimate: "Estimated cost: $X.XX for ~Y tokens"
   
   B. For Claude (Anthropic):
   - Use: "claude-3-5-sonnet-20241022" (200k context)
   - Show model info: "Using Claude 3.5 Sonnet (200k context)"
   - Display cost estimate
   
   C. For Ollama:
   - Warn if local model context is too small
   - Suggest models with large context (llama3.1-70b, mixtral-8x7b)
   - Show: "⚠️ Ensure your Ollama model supports ~X tokens"

4. ADD MISTRAL AI AS CORRECTION OPTION:

   Add "Mistral AI" to the LLM provider dropdown (alongside Claude, OpenAI, Ollama)
   
   When "Mistral AI" is selected:
   
   A. API Key handling (same hybrid approach):
   - Check MISTRAL_API_KEY environment variable
   - If not found, show password input
   
   B. Model selection based on mode:
   - Message-by-Message: "mistral-small-latest" (32k context, cheap)
   - Full Transcript: "mistral-large-latest" (128k context, better quality)
   
   C. Implementation:
   ```python
   from mistralai import Mistral
   
   client = Mistral(api_key=mistral_api_key)
   
   # For message-by-message
   response = client.chat.complete(
       model="mistral-small-latest",
       messages=[
           {"role": "system", "content": "Fix transcription errors..."},
           {"role": "user", "content": transcription}
       ]
   )
   
   # For full transcript
   response = client.chat.complete(
       model="mistral-large-latest",
       messages=[
           {"role": "system", "content": "Review and correct transcript..."},
           {"role": "user", "content": full_transcript}
       ]
   )
   ```
   
   D. Pricing display:
   - mistral-small: ~$0.002 per 1k tokens
   - mistral-large: ~$0.008 per 1k tokens
   - Show estimate: "Estimated cost: $X.XX"

5. UI UPDATES:

   Configuration Section should look like:
   ```
   ⚙️ Configuration
   
   Transcription Engine: [Faster-Whisper ▼]
   Language: [Auto-detect ▼]
   Whisper Model: [base ▼]
   Output Format: [Markdown ▼]
   
   ✨ LLM Correction (Optional)
   ☑️ Enable transcription correction
   
   Provider: [Claude API ▼ OpenAI API / Mistral AI / Ollama Local]
   API Key: [✓ Loaded from environment]
   
   Correction Mode:
   ○ Message-by-Message (Slower, any model)
   ● Full Transcript (Faster, needs large context)
   
   ℹ️ Estimated tokens: ~15,234 tokens
   ℹ️ Using: claude-3-5-sonnet (200k context)
   ℹ️ Estimated cost: ~$0.23
   ```

6. ERROR HANDLING:

   - Token count too high → Force message-by-message mode
   - API rate limits → Show retry countdown
   - Context window exceeded → Clear error, suggest switching modes
   - Parsing errors → Fall back to uncorrected transcript with warning
   - Model not available → Suggest alternative model

7. PROGRESS INDICATORS:

   Message-by-Message mode:
   - "✨ Correcting transcription X of Y..."
   
   Full Transcript mode:
   - "✨ Sending full transcript to LLM (~X tokens)..."
   - "✨ Processing correction response..."
   - "✨ Parsing corrected messages..."

8. PROCESSING SUMMARY UPDATES:

   Add to summary:
   - Correction mode used: "Full Transcript" or "Message-by-Message"
   - Model used: "claude-3-5-sonnet-20241022"
   - Tokens processed: "~15,234 tokens"
   - API cost: "$0.23"
   - Messages corrected: "X of Y successful"

9. DEPENDENCIES:

   Add to requirements.txt (if not already there):
   ```
   mistralai>=1.0.0
   tiktoken>=0.5.0  # for OpenAI token counting
   anthropic>=0.7.0
   ```

10. TOKEN ESTIMATION IMPLEMENTATION:

    Create helper function:
    ```python
    def estimate_tokens(text, provider="claude"):
        """
        Rough token estimation.
        Claude/Anthropic: ~3.5 chars per token
        OpenAI: use tiktoken library
        Mistral: ~4 chars per token
        """
        if provider == "openai":
            import tiktoken
            encoder = tiktoken.encoding_for_model("gpt-4")
            return len(encoder.encode(text))
        elif provider == "claude":
            return len(text) // 3.5
        elif provider == "mistral":
            return len(text) // 4
        else:
            return len(text) // 4  # default estimate
    ```

11. UPDATE README:

    Add section about correction modes:
    ```markdown
    ### LLM Correction Modes
    
    #### Message-by-Message (Default)
    - Corrects each voice message transcription individually
    - Slower but works with any model
    - Better for very long conversations
    - Lower token usage per request
    
    #### Full Transcript (Recommended)
    - Sends entire conversation to LLM at once
    - Faster and more cost-effective
    - Better context for corrections
    - Requires large context window model (100k+ tokens)
    - Shows token estimate before processing
    
    #### Supported LLM Providers
    
    | Provider | Message-by-Message Model | Full Transcript Model | Context |
    |----------|-------------------------|---------------------|---------|
    | Claude | claude-3-5-sonnet | claude-3-5-sonnet | 200k |
    | OpenAI | gpt-4o-mini | gpt-4o | 128k |
    | Mistral | mistral-small | mistral-large | 128k |
    | Ollama | Any local model | Large context models | Varies |
    ```

IMPORTANT NOTES:

- Keep message-by-message as default (safer, works everywhere)
- Make token estimates visible BEFORE processing starts
- Handle context window errors gracefully
- Show clear cost estimates for all providers
- Allow user to cancel if cost/tokens too high
- Test with both small and large conversations
- Ensure date filter updates work smoothly

Please implement all these improvements while maintaining existing functionality and code quality