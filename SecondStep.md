```
Please make two additions to the WhatsApp transcriber app. Implement them one at a time in this order:

ADDITION 1: LANGUAGE SELECTION

Add a language selection feature to the app:

1. Add a dropdown in the Configuration Section with these options:
   - "Auto-detect (Recommended)" - default
   - "Italian"
   - "English" 
   - "Spanish"
   - "French"
   - "German"
   - "Portuguese"
   - "Dutch"
   - "Russian"
   - "Chinese"
   - "Japanese"

2. When calling faster-whisper's transcribe() method, pass the language parameter:
   - If "Auto-detect": don't pass language parameter (let Whisper detect)
   - Otherwise: pass language code (it, en, es, fr, de, pt, nl, ru, zh, ja)

3. Implementation:
   ```python
   # If auto-detect
   segments, info = model.transcribe(audio_file)
   
   # If specific language chosen
   segments, info = model.transcribe(audio_file, language="it")
   ```

4. Show the detected language in the progress text when auto-detect is used:
   "🎤 Transcribing audio X of Y: [filename] (detected: Italian)"

5. Add language info to processing summary and output file headers

ADDITION 2: DATE RANGE FILTERING

Add a date range filtering feature:

1. AFTER ZIP FILE IS UPLOADED AND PARSED:
   - Analyze all messages and audio files to find the date range
   - Extract dates from:
     * Text messages (from chat file timestamps)
     * Audio files (from PTT-YYYYMMDD-WA#### filenames)
   - Determine the full conversation date range (earliest to latest)

2. DISPLAY DATE RANGE INFORMATION:
   - Show a summary box with:
     * "Conversation Date Range: [earliest_date] to [latest_date]"
     * "Total messages: X text, Y audio, Z media files"
   - Make this clearly visible after upload, before configuration

3. ADD DATE RANGE SELECTOR:
   - Add a new section: "Filter by Date Range (Optional)"
   - Include:
     * Checkbox: "Process entire conversation" (checked by default)
     * When unchecked, show two date inputs:
       - Start date (date picker, default = earliest date in conversation)
       - End date (date picker, default = latest date in conversation)
     * Show live count of filtered items:
       "Will process: X text messages, Y audio files, Z media files"
   - Validate that end date >= start date

4. ADD QUICK PRESETS (optional buttons):
   - [Last 7 Days]
   - [Last 30 Days]
   - [Last 3 Months]
   - [Last Year]
   - [Custom Range]
   These automatically set the date range when clicked

5. FILTERING LOGIC:
   - When date range is specified:
     * Only transcribe audio files within date range
     * Only include text messages within date range
     * Only include media files within date range
     * Show skipped items in processing log: "Skipped X items outside date range"
   
6. UPDATE PROGRESS INDICATORS:
   - Progress should reflect filtered count, not total
   - Example: "Transcribing audio 2 of 5 (15 skipped due to date filter)"

7. OUTPUT FILES:
   - Timeline should only contain messages within selected date range
   - Add header note in timeline file:
     "Date range filter applied: YYYY-MM-DD to YYYY-MM-DD"
     "Items outside range were excluded"
   - Media folder should only contain media from selected date range

8. UI PLACEMENT:
   - Insert this section AFTER "Upload Section"
   - BEFORE "Configuration Section"
   - Make it collapsible/expandable
   - Show warning icon if date range is restricted

9. EXAMPLE UI FLOW:
   ```
   📤 Upload Section
      [ZIP file uploaded ✓]
   
   📅 Date Range Filter
      ℹ️ Conversation spans: 01/02/2025 to 10/10/2025
      Total: 150 messages, 45 audio files, 30 media files
      
      ☑️ Process entire conversation
      
      Or filter by date:
      Start: [01/02/2025] 📅
      End:   [10/10/2025] 📅
      
      Quick filters: [Last 7 Days] [Last 30 Days] [Last 3 Months]
      
      → Will process: 150 messages, 45 audio, 30 media
   
   ⚙️ Configuration Section
      Language: [Auto-detect (Recommended) ▼]
      [Whisper model, etc.]
   ```

10. ERROR HANDLING:
    - Handle audio files with unparseable dates gracefully
    - If date is in filename but invalid, include with warning
    - If no dates can be extracted, show warning but allow processing

11. EDGE CASES:
    - Single-day conversations: show both dates as same day
    - Very old chats: handle dates from years ago
    - Future dates (if phone clock was wrong): allow but show warning

TECHNICAL NOTES:
- Parse dates from WhatsApp format in chat file
- Extract dates from audio filenames: PTT-YYYYMMDD-WA####.opus
- Use datetime objects for comparison
- Store filtered items list before processing starts
- Make date filtering efficient (don't load/parse excluded files)
- Maintain existing UI structure and mobile-friendly design
- Add both features to the processing summary output

Please implement both additions maintaining code quality and user experience.
```