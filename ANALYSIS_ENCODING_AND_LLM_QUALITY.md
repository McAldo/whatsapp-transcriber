# Analysis: Encoding Issues & LLM Correction Quality

## Issues Identified

1. **Character Encoding Corruption**: Italian accents (è, à, ò, ù, etc.) are being replaced with corrupted character sequences like `\u00c3\u00a8` or displayed as `�`
2. **LLM Correction Quality**: Automated correction not meeting quality expectations
3. **Identical Timelines**: Both "corrected" and "uncorrected" timelines appear identical

---

## Issue 1: Character Encoding Corruption

### Evidence

**From checkpoint file (0aaea95cfed6437f_checkpoint.json):**
```
PTT-20250226-WA0001.opus: "quando lui � stato"  (should be: "è")
PTT-20250226-WA0002.opus: "S�, in ultimo"      (should be: "Sì")
```

The `�` character (U+FFFD - Unicode replacement character) indicates that:
- Bytes were correctly written as UTF-8
- But then incorrectly read/interpreted as a different encoding (likely ISO-8859-1 or Windows-1252)
- This creates a **double-encoding** issue

### Root Cause Analysis

**The encoding chain:**
```
Voxtral API → Python string → Checkpoint JSON → Timeline generation
```

**Where corruption occurs:**

1. **Voxtral API Response** (Most Likely)
   - Location: `utils/voxtral_transcriber.py:154`
   - Code: `'text': transcription.text`
   - **Issue**: The Voxtral SDK may be returning text with incorrect encoding
   - The Voxtral API might be sending UTF-8 bytes but the SDK is interpreting them as Latin-1

2. **HTTP Response Handling** (Possible)
   - The underlying HTTP library might not be declaring UTF-8 charset
   - If response headers don't specify `Content-Type: application/json; charset=utf-8`
   - Python's requests library may default to ISO-8859-1 (per HTTP spec)

3. **Not the checkpoint** (Confirmed)
   - Checkpoint saving: Uses `encoding='utf-8'` and `ensure_ascii=False` ✓
   - Checkpoint loading: Uses `encoding='utf-8'` ✓
   - The corruption is already present when saved to checkpoint

4. **Not the timeline generation** (Confirmed)
   - Timeline files written with `encoding='utf-8'` ✓
   - The corrupted data is just being passed through

### Technical Details: UTF-8 Double-Encoding

**Example: The letter "è" (e with grave)**

1. Correct UTF-8 encoding: `0xC3 0xA8` (2 bytes)
2. If misinterpreted as ISO-8859-1:
   - `0xC3` → `Ã` (Latin capital A with tilde)
   - `0xA8` → `¨` (diaeresis)
   - Result: "Ã¨"
3. When re-encoded to UTF-8:
   - `Ã` → `\u00c3`
   - `¨` → `\u00a8`
   - Result: The string literal `\u00c3\u00a8` or display as `�`

### Affected Components

**Italian language** is particularly affected because it uses many accented characters:
- à, è, é, ì, ò, ù
- All will show as `�` or escape sequences

**All transcriptions in checkpoint** show this issue, suggesting:
- It happens consistently during transcription
- Not intermittent or random

---

## Issue 2: LLM Correction Quality

### Why Correction Quality is Low

**1. Corrupted Input Text**
```python
# LLM receives:
"quando lui � stato"
# Instead of:
"quando lui è stato"
```
- LLMs are trained on proper UTF-8 text
- Corrupted characters confuse the model
- Model may try to "fix" the corruption, making it worse
- Model can't apply proper Italian grammar corrections

**2. Message-by-Message Mode Lacks Context**
- Each voice message is corrected in isolation
- No context from previous messages
- Harder to resolve ambiguities
- Can't maintain conversation flow

**3. Prompt Design**
- Current prompt (lines 16-30 in `llm_corrector.py`):
  - Generic: Works for any language but not optimized for Italian
  - Minimal context: Doesn't mention conversation context
  - No specific Italian grammar rules

**4. Model Selection**
- Message-by-message uses smaller models (gpt-4o-mini, mistral-small)
- Smaller models have less capability for nuanced corrections
- May not be as strong in Italian language

### Why Both Timelines Appear Identical

**Expected behavior:**
- Uncorrected: Raw transcription with errors
- Corrected: LLM-improved text

**Possible causes:**

1. **Corruption Makes Text Unrecognizable**
   - LLM sees gibberish (`�` characters)
   - Returns input unchanged because it doesn't understand it
   - Result: No actual corrections applied

2. **LLM Changes Are Subtle**
   - Corrections might be minimal (punctuation, capitalization)
   - With corrupted characters overshadowing everything
   - Hard to notice actual corrections

3. **Transcription Already Good**
   - Voxtral might produce high-quality Italian transcriptions
   - Little room for improvement (except for the encoding issue)
   - LLM correctly keeps good transcriptions unchanged

---

## Solutions Proposed

### Solution 1: Fix Character Encoding (HIGH PRIORITY)

**Option A: Fix Voxtral SDK Response Handling**
```python
# In voxtral_transcriber.py, after getting response
def _fix_encoding(text: str) -> str:
    """Fix double-encoded UTF-8 text."""
    try:
        # If text was decoded as Latin-1 but is actually UTF-8
        # Encode back to Latin-1 bytes, then decode as UTF-8
        return text.encode('latin-1').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        # If that fails, text is probably already correct
        return text

# Apply after getting transcription:
result = {
    'text': _fix_encoding(transcription.text),
    # ...
}
```

**Option B: Force UTF-8 in HTTP Client**
```python
# When initializing Voxtral client, ensure response encoding
# This depends on the underlying HTTP library being used
# May need to patch the Voxtral SDK or use a wrapper
```

**Option C: Post-Process Checkpoint**
```python
# Create script to fix existing checkpoint:
def fix_checkpoint_encoding(checkpoint_path):
    """Fix encoding in existing checkpoint file."""
    cp = json.load(open(checkpoint_path, 'r', encoding='utf-8'))

    for filename, trans in cp['transcriptions'].items():
        if 'text' in trans and trans['text']:
            trans['text'] = _fix_encoding(trans['text'])
        if 'corrected_text' in trans and trans['corrected_text']:
            trans['corrected_text'] = _fix_encoding(trans['corrected_text'])

    json.dump(cp, open(checkpoint_path, 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
```

**Testing:**
- Process a new Italian voice message
- Check if accents display correctly
- Verify in checkpoint JSON and timeline output

---

### Solution 2: Improve LLM Correction Quality

**Option A: Enhance Prompt for Italian**
```python
ITALIAN_CORRECTION_PROMPT = """You are correcting automatic transcriptions of Italian voice messages from WhatsApp.

Italian-specific guidelines:
- Fix speech-to-text errors common in Italian (e.g., "ce/c'è", "qual'è/qual è")
- Correct articles and prepositions (il/lo, a/ad, etc.)
- Fix verb conjugations if clearly wrong
- Preserve regional dialects and colloquialisms (it's spoken Italian)
- Add proper punctuation
- Fix capitalization

Original transcription:
{transcription}

Provide only the corrected Italian text:"""
```

**Option B: Add Context to Message-by-Message**
```python
# When correcting message N, include previous 2-3 messages for context
CONTEXT_PROMPT = """You are correcting voice message transcriptions in an ongoing WhatsApp conversation.

Previous messages for context:
{previous_messages}

Voice message to correct:
{transcription}

Correct only the current voice message, but use context to resolve ambiguities:"""
```

**Option C: Improve Bulk Mode Parsing**
```python
# Current issue: Bulk mode parsing is fragile (line 1442)
# Enhancement: Use more robust parsing with fallbacks
def parse_corrected_transcript(corrected: str, original_messages: dict) -> dict:
    """
    More robust parsing of bulk-corrected transcript.
    Falls back to original if parsing fails.
    """
    # Try structured parsing first
    # Fall back to fuzzy matching
    # Return original if uncertain
```

**Option D: Use Better Models for Italian**
```python
# Current: gpt-4o-mini for message-by-message
# Improvement: Use full gpt-4o for Italian (better multilingual)
# OR: Use Claude 3.5 Sonnet (excellent with Italian)

if language == 'it':
    model = "gpt-4o"  # Use better model for Italian
else:
    model = "gpt-4o-mini"
```

---

### Solution 3: Quality Metrics & Validation

**Add automatic quality checks:**
```python
def validate_correction(original: str, corrected: str) -> dict:
    """Check if correction is reasonable."""
    return {
        'length_change': len(corrected) / len(original),
        'has_encoding_issues': '�' in corrected or '\\u00' in corrected,
        'is_empty': len(corrected.strip()) == 0,
        'too_different': len(corrected) < len(original) * 0.5,  # 50% shrinkage = suspicious
        'recommendation': 'keep_original' if any_issue else 'use_corrected'
    }
```

**Show quality metrics in UI:**
```python
st.info(f"""
LLM Correction Summary:
- Corrections applied: 185/209
- Avg length change: +5.2%
- Encoding issues detected: 0
- Rejected corrections: 3 (kept original)
""")
```

---

### Solution 4: User Control & Transparency

**Add timeline comparison view:**
```python
# Generate side-by-side comparison
def generate_comparison_html(messages, transcriptions):
    """Generate HTML with original vs corrected comparison."""
    # Show differences highlighted
    # Allow user to see what changed
```

**Add correction confidence:**
```python
# For each correction, track confidence
trans_result['correction_confidence'] = calculate_confidence(original, corrected)
# Show in timeline: "⚠️ Low confidence correction"
```

**Allow selective correction:**
```python
# UI option: "Only correct if confidence > 80%"
# Otherwise keep original
```

---

## Recommended Implementation Order

### Phase 1: Fix Encoding (Critical - Do First)
1. Implement Option A: Post-decode encoding fix in Voxtral transcriber
2. Create Option C: Script to repair existing checkpoint
3. Test with new Italian voice messages
4. Regenerate timeline from repaired checkpoint

**Priority: CRITICAL** - Everything else depends on this

### Phase 2: Validate Correction Works
1. With encoding fixed, test LLM correction on clean text
2. Verify corrected != uncorrected
3. Manually review sample corrections for quality

### Phase 3: Improve Quality (If Still Needed)
1. Implement Italian-specific prompt (Option A)
2. Add context to message-by-message (Option B)
3. Use better model for Italian (Option D)

### Phase 4: Add Quality Controls
1. Implement validation function
2. Add quality metrics to UI
3. Add comparison view

---

## Testing Checklist

After implementing fixes:

- [ ] Process Italian voice message with accents (è, à, ò, ù)
- [ ] Check checkpoint JSON: Accents display correctly
- [ ] Check timeline output: Accents display correctly
- [ ] Enable LLM correction
- [ ] Verify corrected timeline differs from uncorrected
- [ ] Manually review 10 random corrections for quality
- [ ] Check English transcriptions still work
- [ ] Test bulk mode vs message-by-message
- [ ] Verify costs are reasonable

---

## Current State Summary

**Working:**
- ✅ Checkpoint system functions correctly
- ✅ Transcription completes successfully
- ✅ Timeline generation produces output
- ✅ UTF-8 file I/O is correct

**Broken:**
- ❌ Character encoding in Italian text (shows � or \u00XX)
- ❌ LLM correction produces little/no improvement
- ❌ Both timelines appear identical

**Root Cause:**
- Encoding corruption in Voxtral API response handling
- This cascades to make LLM corrections ineffective

**Fix Strategy:**
- Fix encoding first (enables everything else)
- Then assess if LLM quality needs improvement
- Add quality controls for transparency
