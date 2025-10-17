# Implementation Prompt for Claude Code: Checkpoint & Resume System

## Context
WhatsApp Transcriber app that processes voice messages via Mistral AI API. Currently hangs after ~30-40 files, wasting money on repeated transcriptions. Need resilient checkpoint/resume system to preserve progress and continue after failures.

## Core Requirements

### 1. Checkpoint Manager System
Create `utils/checkpoint_manager.py` with:

**Checkpoint Data Structure:**
```json
{
  "conversation_id": "unique_chat_identifier",
  "chat_name": "conversation_name",
  "timestamp": "2025-01-12T14:30:00",
  "total_files": 100,
  "processed_files": 40,
  "failed_files": 2,
  "transcriptions": {
    "file1.opus": {
      "success": true,
      "text": "transcribed text",
      "corrected_text": "llm corrected text",
      "language": "it",
      "duration": 1.5,
      "cost": 0.0015,
      "timestamp": "2025-01-12T14:25:00",
      "file_size_bytes": 125000,
      "audio_duration_seconds": 45.3
    },
    "file2.opus": {
      "success": false,
      "error": "timeout after 180s",
      "timestamp": "2025-01-12T14:28:30",
      "file_size_bytes": 98000,
      "attempts": 1
    }
  },
  "processing_log": [
    {"time": "2025-01-12T14:25:00", "event": "started", "file": "file1.opus"},
    {"time": "2025-01-12T14:25:15", "event": "transcription_complete", "file": "file1.opus", "duration": 15.2},
    {"time": "2025-01-12T14:25:15", "event": "llm_correction_start", "file": "file1.opus"},
    {"time": "2025-01-12T14:25:22", "event": "llm_correction_complete", "file": "file1.opus", "duration": 7.1},
    {"time": "2025-01-12T14:28:30", "event": "timeout", "file": "file2.opus", "waited": 180}
  ],
  "stats": {
    "total_transcription_cost": 0.045,
    "total_llm_cost": 0.012,
    "total_duration_minutes": 25.3
  }
}
```

**Key Functions:**
- `create_checkpoint(chat_data, checkpoint_dir='chat_checkpoints')` - Initialize new checkpoint
- `save_checkpoint(checkpoint_data, checkpoint_path)` - Save after each file
- `load_checkpoint(checkpoint_path)` - Load existing checkpoint
- `find_checkpoint_for_chat(chat_name, checkpoint_dir)` - Find existing checkpoint by chat name
- `get_pending_files(checkpoint, all_files)` - Return list of unprocessed files
- `add_transcription_result(checkpoint, filename, result, file_info)` - Add result to checkpoint
- `add_log_entry(checkpoint, event_type, filename, details)` - Add to processing log
- `cleanup_old_checkpoints(checkpoint_dir, days=7)` - Auto-cleanup old checkpoints

### 2. Resilient Processor
Create `utils/resilient_processor.py` with:

**Timeout Wrapper:**
- `process_with_timeout(func, args, timeout_seconds=180)` - Wraps transcription with configurable timeout
- Uses threading or multiprocessing to enforce timeout
- Returns `{"success": False, "error": "timeout"}` if exceeded
- Logs detailed timing information

**File Info Collection:**
- Before processing, collect: file size (bytes), audio duration (if possible via librosa/soundfile)
- Add to checkpoint for analysis

**Retry Logic:**
- Track attempt count per file
- Don't retry during main pass (just skip and continue)
- Provide separate "Retry Failed Files" function

### 3. Modified Processing Pipeline (`app.py`)

**On Chat Upload:**
1. Generate conversation ID (hash of chat name + date range)
2. Check for existing checkpoint: `find_checkpoint_for_chat()`
3. If found, show in UI:
   ```
   ⚠️ Previous Progress Found
   - Processed: 40/100 files
   - Failed: 2 files
   - Last run: 2025-01-12 14:30
   
   [Resume Processing] [Start Fresh] [View Existing Results]
   ```

**Processing Logic:**
```python
def process_chat_with_checkpoints(uploaded_file, config, checkpoint=None):
    # Initialize or load checkpoint
    if checkpoint is None:
        checkpoint = create_checkpoint(chat_data)
    
    # Get pending files
    pending_files = get_pending_files(checkpoint, all_audio_files)
    
    # Process each file with timeout
    for audio_file in pending_files:
        # Log start
        add_log_entry(checkpoint, "started", filename)
        
        # Get file info
        file_info = {
            "size_bytes": os.path.getsize(audio_file),
            "duration_seconds": get_audio_duration(audio_file)  # implement safely
        }
        
        # Transcribe with timeout (configurable, default 180s)
        result = process_with_timeout(
            transcriber.transcribe_file,
            (audio_file, config['language']),
            timeout_seconds=config.get('file_timeout', 180)
        )
        
        # Log transcription result
        add_log_entry(checkpoint, 
                     "transcription_complete" if result['success'] else "transcription_failed",
                     filename, 
                     {"duration": elapsed, "error": result.get('error')})
        
        # LLM correction (if enabled)
        if config['use_llm'] and result['success']:
            add_log_entry(checkpoint, "llm_correction_start", filename)
            # ... correction logic
            add_log_entry(checkpoint, "llm_correction_complete", filename, {"duration": elapsed})
        
        # Add to checkpoint
        add_transcription_result(checkpoint, filename, result, file_info)
        
        # Save checkpoint after each file
        save_checkpoint(checkpoint, checkpoint_path)
        
        # Update UI progress
        update_progress_ui(len(checkpoint['transcriptions']), len(all_audio_files))
    
    # Return results (including partial)
    return checkpoint
```

**Auto-Resume Logic:**
```python
if config.get('auto_resume', False):
    # Automatically resume without prompting
    checkpoint = load_checkpoint(found_checkpoint_path)
    return process_chat_with_checkpoints(file, config, checkpoint)
else:
    # Show UI prompt with resume/fresh/view options
    show_resume_prompt_ui(found_checkpoint_path)
```

### 4. UI Changes (`app.py`)

**New Settings Section:**
```python
with st.expander("⚙️ Processing Settings", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        file_timeout = st.number_input(
            "File Timeout (seconds)",
            min_value=60,
            max_value=600,
            value=180,
            step=30,
            help="Maximum time to wait per file before skipping"
        )
        
        auto_resume = st.checkbox(
            "Auto-resume interrupted processing",
            value=False,
            help="Automatically continue from last checkpoint without prompting"
        )
    
    with col2:
        save_uncorrected = st.checkbox(
            "Save uncorrected timeline",
            value=True,
            help="Save both corrected and uncorrected versions for comparison"
        )
```

**Default Model Settings:**
```python
# Set defaults
transcription_engine = st.selectbox(
    "Transcription Engine",
    ['voxtral', 'faster-whisper'],
    index=0,  # Voxtral (Mistral AI) as default
    help="Voxtral Mini via Mistral AI recommended for best quality"
)

if config.get('use_llm', False):
    llm_provider = st.selectbox(
        "LLM Provider",
        ['openai', 'claude', 'mistral', 'ollama'],
        index=0,  # OpenAI as default
        help="OpenAI recommended for best correction quality"
    )
```

**Resume UI (when checkpoint found):**
```python
if checkpoint_found:
    st.warning("⚠️ Previous processing found for this chat")
    
    checkpoint = load_checkpoint(checkpoint_path)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Processed", f"{checkpoint['processed_files']}/{checkpoint['total_files']}")
    col2.metric("Failed", checkpoint['failed_files'])
    col3.metric("Last Run", checkpoint['timestamp'][:16])
    
    col_a, col_b, col_c = st.columns(3)
    
    if col_a.button("▶️ Resume Processing"):
        st.session_state['resume_checkpoint'] = checkpoint
        st.session_state['processing_mode'] = 'resume'
        st.rerun()
    
    if col_b.button("🔄 Start Fresh"):
        os.remove(checkpoint_path)  # Delete checkpoint
        st.session_state['processing_mode'] = 'fresh'
        st.rerun()
    
    if col_c.button("📄 View Results"):
        # Generate timeline from checkpoint data
        generate_timeline_from_checkpoint(checkpoint)
```

**Progress Display During Processing:**
```python
progress_bar = st.progress(0)
status_text = st.empty()
stats_cols = st.columns(4)

# Update in processing loop
progress = processed / total
progress_bar.progress(progress)
status_text.text(f"Processing: {current_file} ({processed}/{total})")
stats_cols[0].metric("Completed", processed)
stats_cols[1].metric("Failed", failed)
stats_cols[2].metric("Cost", f"${total_cost:.4f}")
stats_cols[3].metric("Time", f"{elapsed_time:.1f}s")
```

**Retry Failed Files Button:**
```python
if checkpoint['failed_files'] > 0:
    if st.button("🔁 Retry Failed Files"):
        failed_file_list = [f for f, data in checkpoint['transcriptions'].items() 
                           if not data.get('success', False)]
        # Process only failed files
        retry_failed_files(checkpoint, failed_file_list, config)
```

### 5. Timeline Generation Updates

**Generate Both Versions:**
```python
def generate_timelines(messages, transcriptions, config, filter_stats):
    """Generate both corrected and uncorrected timelines."""
    
    # Always generate corrected timeline
    corrected_timeline = generate_timeline(
        messages, 
        transcriptions,  # Uses corrected_text if available
        config['output_format'],
        filter_stats
    )
    
    uncorrected_timeline = None
    if config.get('save_uncorrected', True):
        # Generate uncorrected version
        uncorrected_transcriptions = {
            k: {**v, 'corrected_text': None}  # Remove corrected text
            for k, v in transcriptions.items()
        }
        uncorrected_timeline = generate_timeline(
            messages,
            uncorrected_transcriptions,
            config['output_format'],
            filter_stats
        )
    
    return corrected_timeline, uncorrected_timeline
```

**Save Both Files:**
```python
# In file packaging
if uncorrected_timeline:
    uncorrected_filename = output_filename.replace('.md', '_uncorrected.md')
    # Save both to output ZIP
```

### 6. Detailed Logging

**Create Processing Log:**
```python
# In checkpoint, maintain detailed log
processing_log = [
    {
        "timestamp": "2025-01-12T14:25:00.123",
        "event": "transcription_start",
        "file": "PTT-20250424-WA0000.opus",
        "file_size_bytes": 125000,
        "audio_duration_seconds": 45.3
    },
    {
        "timestamp": "2025-01-12T14:25:15.456",
        "event": "transcription_complete",
        "file": "PTT-20250424-WA0000.opus",
        "duration_seconds": 15.2,
        "text_length": 2500,
        "cost": 0.0015
    },
    {
        "timestamp": "2025-01-12T14:28:30.789",
        "event": "timeout",
        "file": "PTT-20250424-WA0001.opus",
        "timeout_after_seconds": 180,
        "file_size_bytes": 98000
    }
]
```

**Export Log Function:**
```python
def export_processing_log(checkpoint, output_path):
    """Export detailed processing log as CSV for analysis."""
    import pandas as pd
    
    df = pd.DataFrame(checkpoint['processing_log'])
    df.to_csv(output_path, index=False)
    
    # Also generate summary statistics
    summary = analyze_processing_log(checkpoint)
    return summary
```

**UI to Download Log:**
```python
if st.button("📊 Download Processing Log"):
    log_csv = export_processing_log(checkpoint, 'processing_log.csv')
    st.download_button(
        "Download Log CSV",
        log_csv,
        "processing_log.csv",
        "text/csv"
    )
```

### 7. Helper Utilities

**Get Audio Duration (safe):**
```python
def get_audio_duration(audio_path):
    """Safely get audio duration in seconds."""
    try:
        # Try with soundfile first (lightweight)
        import soundfile as sf
        info = sf.info(audio_path)
        return info.duration
    except:
        try:
            # Fallback to librosa
            import librosa
            duration = librosa.get_duration(path=audio_path)
            return duration
        except:
            # If all fails, return None
            return None
```

**Timeout Wrapper (threading-based):**
```python
import threading
from typing import Callable, Any, Tuple

def process_with_timeout(func: Callable, args: Tuple, timeout_seconds: int) -> dict:
    """Execute function with timeout using threading."""
    result = {"success": False, "error": "timeout"}
    
    def target():
        nonlocal result
        try:
            result = func(*args)
        except Exception as e:
            result = {"success": False, "error": str(e)}
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout_seconds)
    
    if thread.is_alive():
        # Timeout occurred
        return {"success": False, "error": f"timeout after {timeout_seconds}s"}
    
    return result
```

## Implementation Order

1. **Create `utils/checkpoint_manager.py`** (30 min)
   - Implement all checkpoint save/load/manage functions
   - Add processing log management
   - Test with dummy data

2. **Create `utils/resilient_processor.py`** (15 min)
   - Implement timeout wrapper
   - Implement file info collection
   - Test timeout behavior

3. **Modify `app.py` - Settings Section** (10 min)
   - Add file timeout setting
   - Add auto-resume checkbox
   - Add save uncorrected checkbox
   - Set default models (Voxtral, OpenAI)

4. **Modify `app.py` - Checkpoint Detection** (15 min)
   - Check for existing checkpoint on upload
   - Show resume UI if found
   - Implement resume/fresh/view logic

5. **Modify `app.py` - Processing Pipeline** (30 min)
   - Integrate checkpoint manager
   - Add timeout wrapper to transcription calls
   - Add file info collection
   - Save checkpoint after each file
   - Update progress UI

6. **Modify `app.py` - Timeline Generation** (15 min)
   - Generate both corrected and uncorrected versions
   - Save both to output ZIP
   - Label files clearly

7. **Add Retry Failed Files Feature** (15 min)
   - Button to retry failed files
   - Process only failed files from checkpoint
   - Update checkpoint with results

8. **Add Processing Log Export** (10 min)
   - Button to download log CSV
   - Include summary statistics
   - Format for easy analysis

9. **Testing** (30 min)
   - Test with small chat (10 files)
   - Interrupt processing (Ctrl+C)
   - Verify checkpoint saved
   - Resume and verify continuation
   - Test retry failed files
   - Verify both timeline versions generated

## Files to Create/Modify

**New Files:**
- `utils/checkpoint_manager.py`
- `utils/resilient_processor.py`

**Modified Files:**
- `app.py` (main changes in processing pipeline and UI)
- `utils/voxtral_transcriber.py` (minimal - just add file info collection if needed)

**New Directory:**
- `chat_checkpoints/` (auto-created, add to .gitignore)

## Testing Checklist

- [ ] Checkpoint created on first file processed
- [ ] Checkpoint updated after each file
- [ ] Processing can be interrupted (Ctrl+C) and resumed
- [ ] Failed files marked in checkpoint
- [ ] Retry failed files works
- [ ] Both timelines generated when option enabled
- [ ] Processing log exported correctly
- [ ] Auto-resume works when enabled
- [ ] Manual resume prompt works when disabled
- [ ] Old checkpoints cleaned up after 7 days
- [ ] File timeout works (artificially create slow file)
- [ ] Progress preserved across app restarts
- [ ] Works from mobile phone (Tailscale)

## Important Notes

- **Checkpoint file location:** `chat_checkpoints/{conversation_id}_checkpoint.json`
- **Conversation ID generation:** Use hash of (chat_name + date_range + file_count) for uniqueness
- **Thread safety:** Use file locking when saving checkpoint if needed
- **Error handling:** All checkpoint operations should fail gracefully
- **Performance:** Checkpoint save should be fast (<100ms)
- **Mobile compatibility:** Checkpoint should work when accessing via phone
- **Cost tracking:** Maintain running totals in checkpoint

## Success Criteria

After implementation:
1. ✅ Can interrupt processing at any point without losing progress
2. ✅ Can resume from last processed file
3. ✅ Failed files don't block processing
4. ✅ Detailed log available for debugging
5. ✅ Both timeline versions available for comparison
6. ✅ No money wasted on repeated transcriptions
7. ✅ User can walk away and return later to resume

## Additional Requirements

- Add `soundfile` or `librosa` to `requirements.txt` for audio duration detection
- Update `.gitignore` to exclude `chat_checkpoints/`
- Add brief documentation in README about checkpoint feature
- Consider adding checkpoint cleanup command to UI (clear old checkpoints)

---

**Estimated Total Implementation Time:** 2.5-3 hours
**Testing Time:** 30 minutes
**Total:** ~3.5 hours

Ready to implement!
