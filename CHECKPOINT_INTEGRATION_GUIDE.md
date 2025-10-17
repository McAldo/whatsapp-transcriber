# Checkpoint Integration Guide for process_chat()

## Overview

This document provides step-by-step instructions for integrating the checkpoint system into the `process_chat()` function in `app.py`. The checkpoint system is already implemented in `utils/checkpoint_manager.py` and `utils/resilient_processor.py`.

---

## Current Status

✅ **Completed:**
- Checkpoint manager module created
- Resilient processor module created
- UI for checkpoint detection and resume
- Processing settings added
- All supporting infrastructure

⏳ **In Progress:**
- Integrating checkpoints into `process_chat()` function

---

## Modification Strategy

The `process_chat()` function (lines ~967-1370) needs to be modified to:

1. **Accept optional checkpoint parameter**
2. **Initialize or load checkpoint at start**
3. **Determine which files to process** (all vs pending)
4. **Wrap transcription calls with timeout**
5. **Collect file info before processing**
6. **Add log entries for events**
7. **Save checkpoint after each file**
8. **Update progress displays**

---

## Detailed Modifications

### 1. Function Signature

**Current (line 967):**
```python
def process_chat(uploaded_file, config):
```

**Modified:**
```python
def process_chat(uploaded_file, config, checkpoint=None):
    """
    Main processing function with checkpoint support.

    Args:
        uploaded_file: Uploaded ZIP file
        config: Configuration dictionary
        checkpoint: Optional existing checkpoint to resume from

    Returns:
        Results dictionary
    """
```

### 2. Checkpoint Initialization (After line 992)

**Add after progress bar initialization:**

```python
# Progress tracking
progress_bar = st.progress(0)
status_text = st.empty()

# ADD THIS SECTION:
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
```

### 3. Create or Load Checkpoint (After parsing, before transcription - line ~1040)

**Add after filtering is applied:**

```python
status_text.markdown(f"**📝 Processing {len(messages)} messages, {len(audio_files)} audio files**")
progress_bar.progress(30)

# ADD THIS SECTION:
# Initialize or resume checkpoint
if is_resuming:
    status_text.markdown("**📋 Resuming from checkpoint...**")
    # Checkpoint already loaded, just get path
    checkpoint_path = checkpoint_mgr.save_checkpoint(checkpoint)  # Get path
else:
    # Create new checkpoint
    conv_name = extract_conversation_name(uploaded_file.name) if hasattr(uploaded_file, 'name') else "Chat"
    date_range_str = ""
    if filter_stats.get('filtered'):
        date_range_str = f"{filter_stats['start_date']} to {filter_stats['end_date']}"
    else:
        date_range_str = "all"

    chat_data = {
        'conversation_id': checkpoint_mgr.generate_conversation_id(conv_name, date_range_str, len(audio_files)),
        'chat_name': conv_name,
        'total_files': len(audio_files),
        'date_range': date_range_str,
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
if is_resuming:
    pending_audio_files = checkpoint_mgr.get_pending_files(checkpoint, audio_files)
    status_text.markdown(f"**▶️ Resuming: {len(pending_audio_files)} files remaining**")
else:
    pending_audio_files = audio_files

# Update initial metrics
completed_metric.metric("Completed", checkpoint['processed_files'])
failed_metric.metric("Failed", checkpoint['failed_files'])
cost_metric.metric("Cost", f"${checkpoint['stats']['total_transcription_cost']:.4f}")
time_metric.metric("Time", "0s")
```

### 4. Modify Voxtral Transcription Loop (lines ~1069-1094)

**Replace the existing loop:**

```python
# OLD:
for idx, audio_file in enumerate(audio_files, 1):
    filename = os.path.basename(audio_file)
    progress = 30 + int((idx / len(audio_files)) * 40)
    progress_bar.progress(progress)

    status_text.markdown(...)

    result = transcriber.transcribe_file(audio_file, language=lang_param)
    transcriptions[filename] = result

    # ...

# NEW:
for idx, audio_file in enumerate(pending_audio_files, 1):
    filename = os.path.basename(audio_file)

    # Calculate progress based on total files (not just pending)
    total_processed = checkpoint['processed_files'] + idx
    progress = 30 + int((total_processed / checkpoint['total_files']) * 40)
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
    start_time = time.time()

    result = process_with_timeout(
        transcriber.transcribe_file,
        (audio_file,),
        timeout_seconds=config.get('file_timeout', 180),
        kwargs={'language': lang_param}
    )

    elapsed = time.time() - start_time

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
```

### 5. Modify Faster-Whisper Transcription Loop (lines ~1137-1157)

**Apply similar changes as Voxtral:**

```python
# Replace the loop with checkpoint-aware version
for idx, audio_file in enumerate(pending_audio_files, 1):
    filename = os.path.basename(audio_file)

    # Calculate progress
    total_processed = checkpoint['processed_files'] + idx
    progress = 30 + int((total_processed / checkpoint['total_files']) * 40)
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
    start_time = time.time()

    result = process_with_timeout(
        transcriber.transcribe_file,
        (audio_file,),
        timeout_seconds=config.get('file_timeout', 180),
        kwargs={'language': lang_param}
    )

    elapsed = time.time() - start_time

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

    # Save checkpoint
    checkpoint_mgr.save_checkpoint(checkpoint, checkpoint_path)

    # Update metrics
    completed_metric.metric("Completed", f"{checkpoint['processed_files']}/{checkpoint['total_files']}")
    failed_metric.metric("Failed", checkpoint['failed_files'])
    cost_metric.metric("Cost", f"${checkpoint['stats']['total_transcription_cost']:.4f}")
    time_metric.metric("Time", f"{elapsed:.1f}s")

    # Log metrics
    log_processing_metrics(filename, file_info, result, elapsed)

    # Store in transcriptions dict
    transcriptions[filename] = result

    # Update status with detected language
    if config['language'] == 'auto' and result.get('success') and result.get('language'):
        detected_lang = result['language']
        status_text.markdown(
            f"**🎤 Transcribing {total_processed}/{checkpoint['total_files']}: {filename}**\n\n"
            f"Progress: {progress}% | Detected: {detected_lang.upper()}"
        )
```

### 6. Update LLM Correction Section (lines ~1255-1268)

**For message-by-message mode, add checkpoint tracking:**

```python
else:
    # Message-by-message mode (original behavior)
    for idx, (filename, trans_result) in enumerate(transcriptions.items(), 1):
        if trans_result['success'] and trans_result['text']:
            progress = 70 + int((idx / len(transcriptions)) * 10)
            progress_bar.progress(progress)

            status_text.markdown(
                f"**✨ Correcting {idx}/{len(transcriptions)}...**"
            )

            # ADD: Log LLM correction start
            checkpoint_mgr.add_log_entry(checkpoint, "llm_correction_start", filename)

            import time
            start_time = time.time()

            corrected = corrector.correct_transcription(trans_result['text'])
            trans_result['corrected_text'] = corrected
            trans_result['llm_corrected'] = True

            elapsed = time.time() - start_time

            # ADD: Log LLM correction complete
            checkpoint_mgr.add_log_entry(checkpoint, "llm_correction_complete", filename, {
                "duration": elapsed
            })

            # ADD: Save checkpoint (optional - could be done less frequently)
            if idx % 5 == 0:  # Save every 5 corrections
                checkpoint_mgr.save_checkpoint(checkpoint, checkpoint_path)

    # Final save after all corrections
    checkpoint_mgr.save_checkpoint(checkpoint, checkpoint_path)

    status_text.markdown(f"**✨ Applied AI corrections using {config['llm_provider']}**")
```

### 7. Final Checkpoint Save (Before return - line ~1360)

**Add final checkpoint update:**

```python
results['stats'] = {
    # ... existing stats ...
}

# ADD: Final checkpoint save with completion marker
checkpoint['completed'] = True
checkpoint_mgr.save_checkpoint(checkpoint, checkpoint_path)

# ADD: Cleanup old checkpoints
checkpoint_mgr.cleanup_old_checkpoints(days=7)

results['checkpoint_path'] = checkpoint_path  # Include in results
```

---

## Calling the Modified Function

### From main() - Handle Resume Mode (line ~1938)

**Modify the call site:**

```python
# OLD:
results = process_chat(uploaded_file, config)

# NEW:
# Check if resuming
resume_checkpoint = st.session_state.get('resume_checkpoint')
if st.session_state.get('processing_mode') == 'resume' and resume_checkpoint:
    st.info(f"▶️ Resuming from checkpoint: {resume_checkpoint['processed_files']}/{resume_checkpoint['total_files']} files completed")
    results = process_chat(uploaded_file, config, checkpoint=resume_checkpoint)
    # Clear resume state
    st.session_state.resume_checkpoint = None
    st.session_state.processing_mode = None
else:
    results = process_chat(uploaded_file, config)
```

---

## Error Handling Improvements

### Add Try-Catch Around Individual Files

To ensure one file failure doesn't crash the entire process:

```python
for idx, audio_file in enumerate(pending_audio_files, 1):
    try:
        # ... existing transcription code ...
    except Exception as e:
        logger.error(f"Unexpected error processing {filename}: {e}")

        # Create failure result
        result = {
            'success': False,
            'error': f"Unexpected error: {str(e)}",
            'error_type': type(e).__name__
        }

        # Add to checkpoint
        checkpoint_mgr.add_transcription_result(checkpoint, filename, result, file_info)
        checkpoint_mgr.add_log_entry(checkpoint, "error", filename, {"error": str(e)})
        checkpoint_mgr.save_checkpoint(checkpoint, checkpoint_path)

        transcriptions[filename] = result

        # Continue to next file
        continue
```

---

## Testing Checklist

After making these modifications, test:

- [ ] **Fresh processing** - Start new chat, process completely
- [ ] **Checkpoint creation** - Verify checkpoint file created in `chat_checkpoints/`
- [ ] **Interrupt & Resume** - Stop processing (Ctrl+C), restart app, resume
- [ ] **Timeout handling** - Artificially slow file to trigger timeout
- [ ] **Failed file handling** - Corrupt audio file to test error handling
- [ ] **Progress persistence** - Verify metrics/counts persist across resume
- [ ] **Duplicate prevention** - Resume doesn't reprocess completed files
- [ ] **LLM correction tracking** - Verify LLM events logged in checkpoint
- [ ] **Final cleanup** - Old checkpoints removed after 7 days

---

## Quick Reference: Key Functions

```python
# From utils.checkpoint_manager
CheckpointManager()
checkpoint_mgr.create_checkpoint(chat_data)
checkpoint_mgr.save_checkpoint(checkpoint, path)
checkpoint_mgr.load_checkpoint(path)
checkpoint_mgr.get_pending_files(checkpoint, all_files)
checkpoint_mgr.add_transcription_result(checkpoint, filename, result, file_info)
checkpoint_mgr.add_log_entry(checkpoint, event_type, filename, details)

# From utils.resilient_processor
process_with_timeout(func, args, timeout_seconds, kwargs)
get_file_info(file_path)
log_processing_metrics(filename, file_info, result, elapsed)
```

---

## Implementation Time Estimate

- **Modify function signature**: 2 minutes
- **Add checkpoint initialization**: 10 minutes
- **Modify Voxtral loop**: 15 minutes
- **Modify Faster-Whisper loop**: 15 minutes
- **Update LLM section**: 10 minutes
- **Update call site**: 5 minutes
- **Testing**: 20 minutes

**Total**: ~75 minutes

---

## Next Steps

1. Apply modifications to `process_chat()` function in order presented
2. Test with small chat export (3-5 files)
3. Test interrupt/resume functionality
4. Implement remaining features:
   - Generate both timeline versions
   - Retry failed files button
   - Processing log export
5. Update README with checkpoint feature documentation

---

## Notes

- All checkpoint operations are designed to fail gracefully
- File locking not implemented - avoid running multiple instances on same chat
- Checkpoint size grows with number of files (~1KB per file)
- Consider adding checkpoint compression for very large chats (1000+ files)

---

**Status**: Ready for implementation
**Last Updated**: 2025-01-12
