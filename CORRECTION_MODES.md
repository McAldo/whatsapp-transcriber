# LLM Correction Modes

The WhatsApp Transcriber now supports two modes for AI transcription correction:

## 🎯 Correction Modes

### Message-by-Message Mode (Default)
Each voice message is corrected independently by the LLM.

**Benefits:**
- ✅ Works with any LLM context window size
- ✅ Suitable for very long conversations (100+ messages)
- ✅ Lower token usage per API call
- ✅ More predictable cost

**Limitations:**
- ⚠️ Names may be spelled inconsistently across messages
- ⚠️ Less context for ambiguous words
- ⚠️ Requires multiple API calls

### Full Transcript Mode (Recommended)
The LLM sees all voice messages at once in a single request.

**Benefits:**
- ✅ **Consistent name spelling** throughout the conversation
- ✅ **Better context understanding** - LLM sees the conversation flow
- ✅ **More accurate corrections** for ambiguous words
- ✅ **Single API call** instead of many
- ✅ More efficient for conversations under 100 messages

**Limitations:**
- ⚠️ Requires larger context window (may not work with smaller models)
- ⚠️ Higher token usage per request (but often fewer total tokens)
- ⚠️ Not recommended for very long conversations (100+ messages)

## 📝 How It Works

### Message-by-Message
```
Message 1 → LLM → Corrected 1
Message 2 → LLM → Corrected 2
Message 3 → LLM → Corrected 3
...
```

### Full Transcript
```
[1] Message 1
[2] Message 2    →  LLM  →  [1] Corrected 1
[3] Message 3              [2] Corrected 2
...                        [3] Corrected 3
                           ...
```

## 🎨 Example: Name Recognition

### Message-by-Message Mode
```
[1] "Hi, my name is jon smith"
[2] "Nice to meet you john"
[3] "Call me Jhon"
```
Result: Inconsistent - "jon smith", "john", "Jhon"

### Full Transcript Mode
```
[1] "Hi, my name is jon smith"
[2] "Nice to meet you john"
[3] "Call me Jhon"
```
Result: Consistent - "John Smith" throughout

## ⚙️ Configuration

In the app UI under "AI Transcription Enhancement":

1. Enable "Enable AI correction"
2. Select **Correction Mode:**
   - **Message-by-Message**: Safer for large chats
   - **Full Transcript**: Better context, more consistent

## 💰 Cost Considerations

### Message-by-Message
- Cost: N API calls (where N = number of messages)
- Tokens per call: ~100-500
- Example: 20 messages × 200 tokens = 4,000 tokens total

### Full Transcript
- Cost: 1 API call
- Tokens per call: ~2,000-8,000
- Example: 20 messages in single call = 3,000 tokens total

**Note:** Full transcript mode is often more cost-effective for small to medium conversations!

## 🔧 Technical Details

### Implementation
- **File**: `utils/llm_corrector.py`
- **Method**: `correct_full_transcript()`
- **Format**: Messages sent as `[1] text\n[2] text\n...`
- **Parsing**: Regex extracts numbered responses

### Supported Providers
- ✅ Claude (Anthropic) - 8K max_tokens
- ✅ OpenAI (GPT-4o-mini) - 8K max_tokens  
- ✅ Ollama (Local) - 4K num_predict

## 📊 Recommendations

| Conversation Size | Recommended Mode | Reason |
|------------------|------------------|---------|
| < 50 messages | Full Transcript | Best results, most efficient |
| 50-100 messages | Full Transcript | Still works well |
| 100+ messages | Message-by-Message | Safer, won't exceed context limits |
| Very technical terms | Full Transcript | Better consistency |
| Multiple speakers | Full Transcript | Better name recognition |

## 🚀 Usage Example

```python
# In app.py configuration
config = {
    'use_llm': True,
    'llm_provider': 'claude',
    'llm_correction_mode': 'full',  # or 'message'
    ...
}

# Processing handles both modes automatically
if correction_mode == 'full':
    # Single API call with all messages
    corrected_dict = corrector.correct_full_transcript(trans_list)
else:
    # Multiple API calls, one per message
    for message in messages:
        corrected = corrector.correct_transcription(message)
```

## ✅ Testing

The parsing logic has been tested with:
- ✅ Single-line messages
- ✅ Multi-line messages
- ✅ Messages with special characters
- ✅ Edge cases (empty messages, large numbers)

