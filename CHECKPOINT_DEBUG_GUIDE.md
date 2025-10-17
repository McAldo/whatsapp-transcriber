# Checkpoint Detection Debugging Guide

## Your Situation

You processed 185/209 files, got an error, and now when you reload the app and upload the file, you don't see the resume button.

---

## Step 1: Check if Checkpoint Exists

Run this command in your terminal:

```bash
python check_checkpoint.py
```

This will show you:
- All checkpoint files found
- Their status (how many files processed)
- The conversation ID and chat name
- Whether the ID matches what it should be

---

## Step 2: Check the Logs When Uploading

When you upload the ZIP file in the app, watch your terminal. You should see logs like:

```
INFO - Checkpoint detection: conv_name='Rosella Pizzi', files=209, id=9b57608a36a83156
INFO - Exact ID match result: None
INFO - Exact match not found. Searching 1 checkpoint(s) by file count...
INFO - Checking 0aaea95cfed6437f_checkpoint.json: 209 files, name='Rosella Pizzi (2)'
INFO - ✓ Found matching checkpoint by file count: chat_checkpoints/0aaea95cfed6437f_checkpoint.json
```

If you see "✓ Found matching checkpoint", the fallback search is working!

---

## Step 3: Understanding the Problem

The issue is with how the filename is extracted:

**First upload:**
- File: `WhatsApp Chat with Rosella Pizzi.zip`
- Extracted name: `Rosella Pizzi (2)` (the (2) comes from Windows/Streamlit)
- Conversation ID: `0aaea95cfed6437f`
- Checkpoint created: `0aaea95cfed6437f_checkpoint.json` ✓

**Second upload (after error):**
- File: `WhatsApp Chat with Rosella Pizzi.zip`
- Extracted name: `Rosella Pizzi` (no suffix this time!)
- Conversation ID: `9b57608a36a83156` (different!)
- Checkpoint lookup: Failed ✗
- Fallback search: Should find it by file count (209 files) ✓

---

## Step 4: Verify Fallback is Working

After uploading the file, you should see:

1. **In the app UI:** Blue info box saying:
   ```
   📋 Found existing checkpoint by file count match: 0aaea95cfed6437f_checkpoint.json
   ```

2. **Below the file upload:** Warning box showing:
   ```
   ⚠️ Previous processing found for this chat
   Processed: 185/209
   Failed: 0
   Last Run: 2025-10-14 13:51

   [▶️ Resume Processing] [🔄 Start Fresh] [🔁 Retry Failed Files]
   ```

---

## Step 5: If Resume Button Still Doesn't Appear

### Option A: Manual Checkpoint Rename

1. Run `python check_checkpoint.py` to get the current checkpoint ID
2. Calculate the correct ID for your current upload:
   ```bash
   python -c "import hashlib; print(hashlib.sha256('Rosella Pizzi|all|209'.encode()).hexdigest()[:16])"
   ```
3. Rename the checkpoint file to match:
   ```bash
   cd chat_checkpoints
   rename 0aaea95cfed6437f_checkpoint.json 9b57608a36a83156_checkpoint.json
   ```
4. Upload the file again

### Option B: Force Clear Upload Cache

1. Close the Streamlit app
2. Delete browser cache for localhost:8501
3. Restart the app
4. Upload the ZIP file (don't drag the same file object, select it fresh from file picker)

### Option C: Use Exact Same Filename

1. Don't re-select the file from file picker
2. If you must re-upload:
   - Make a copy of the ZIP with a different name first
   - Upload the copy (this forces a clean upload)

---

## Step 6: Check Session State

The checkpoint detection should happen automatically when you upload the file. The key session state variables are:

- `st.session_state.checkpoint_found` - Should be `True`
- `st.session_state.checkpoint_path` - Should have the path
- `st.session_state.uploaded_data` - Should have the file data

If these aren't set, the checkpoint detection code might not be running.

---

## Testing the Fix

1. **Save your current work:** The checkpoint file is safe in `chat_checkpoints/`

2. **Try uploading again:**
   - Close and restart the app
   - Upload the EXACT same ZIP file
   - Watch the terminal for logs

3. **You should see:**
   - Log messages about checkpoint detection
   - Blue info box about found checkpoint
   - Resume button appearing below

4. **If it works:**
   - Click "▶️ Resume Processing"
   - It should continue from file 186/209

---

## Emergency: Can't Get Resume to Work?

If you absolutely can't get the resume button to appear, you can manually resume:

1. Open the checkpoint file:
   ```bash
   notepad chat_checkpoints\0aaea95cfed6437f_checkpoint.json
   ```

2. Look at the `transcriptions` section - it has all 185 completed transcriptions

3. You have two options:
   - **Wait for fix:** Keep the checkpoint safe, we'll debug further
   - **Manual merge:** Extract the successful transcriptions and use them

---

## Prevention for Future

To avoid this issue in the future:

1. **Use stable filenames:** Don't rename the ZIP between runs
2. **Don't re-upload same file:** If app crashes, just restart - the uploaded data should still be in session
3. **Check logs:** Always monitor terminal output for checkpoint detection messages

---

## What We Fixed

1. ✅ Added fallback search by file count (most reliable)
2. ✅ Changed ID generation to use "all" instead of date ranges
3. ✅ Added comprehensive logging for debugging
4. ✅ Added UI message when fallback finds checkpoint
5. ✅ Fixed progress bar and counter issues

---

## Next Steps

1. Run `python check_checkpoint.py` to see checkpoint status
2. Try uploading the file again and watch the logs
3. Report back what you see in the logs
4. If fallback isn't triggering, we'll debug further

The checkpoint with your 185 transcriptions is safe and intact!
