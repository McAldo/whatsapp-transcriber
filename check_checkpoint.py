#!/usr/bin/env python3
"""
Simple script to check checkpoint status
"""

import json
import os
import glob
from datetime import datetime

checkpoint_dir = "chat_checkpoints"

print("=" * 80)
print("CHECKPOINT DIAGNOSTIC TOOL")
print("=" * 80)
print()

if not os.path.exists(checkpoint_dir):
    print(f"❌ Checkpoint directory '{checkpoint_dir}' does not exist!")
    exit(1)

checkpoints = glob.glob(os.path.join(checkpoint_dir, "*_checkpoint.json"))

if not checkpoints:
    print(f"❌ No checkpoint files found in '{checkpoint_dir}'")
    exit(0)

print(f"✓ Found {len(checkpoints)} checkpoint file(s):")
print()

for cp_path in checkpoints:
    print(f"📄 {os.path.basename(cp_path)}")
    print("-" * 80)

    try:
        with open(cp_path, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)

        # Basic info
        print(f"  Conversation ID: {checkpoint['conversation_id']}")
        print(f"  Chat Name: {checkpoint['chat_name']}")
        print(f"  Last Updated: {checkpoint['timestamp']}")
        print()

        # Progress
        total = checkpoint['total_files']
        processed = checkpoint['processed_files']
        failed = checkpoint['failed_files']
        succeeded = processed - failed
        remaining = total - processed

        print(f"  Progress:")
        print(f"    Total Files: {total}")
        print(f"    Processed: {processed} ({processed/total*100:.1f}%)")
        print(f"    Succeeded: {succeeded}")
        print(f"    Failed: {failed}")
        print(f"    Remaining: {remaining}")
        print()

        # Costs
        trans_cost = checkpoint['stats']['total_transcription_cost']
        llm_cost = checkpoint['stats']['total_llm_cost']
        total_cost = trans_cost + llm_cost

        print(f"  Costs:")
        print(f"    Transcription: ${trans_cost:.4f}")
        print(f"    LLM: ${llm_cost:.4f}")
        print(f"    Total: ${total_cost:.4f}")
        print()

        # Configuration
        print(f"  Config:")
        print(f"    Engine: {checkpoint['config'].get('transcription_engine', 'N/A')}")
        print(f"    Use LLM: {checkpoint['config'].get('use_llm', False)}")
        print(f"    LLM Provider: {checkpoint['config'].get('llm_provider', 'N/A')}")
        print()

        # Recent activity
        log_entries = checkpoint.get('processing_log', [])
        if log_entries:
            print(f"  Recent Events (last 5):")
            for entry in log_entries[-5:]:
                event_time = entry.get('time', '')[:19]
                event_type = entry.get('event', 'unknown')
                event_file = entry.get('file', '')
                print(f"    {event_time} - {event_type} - {event_file}")
        else:
            print(f"  No events logged")

        print()

        # Generate test conversation ID
        import hashlib
        test_id = hashlib.sha256(f"{checkpoint['chat_name']}|all|{total}".encode()).hexdigest()[:16]
        print(f"  🔍 Test: Conversation ID for '{checkpoint['chat_name']}' + 'all' + {total} files:")
        print(f"     Expected: {test_id}")
        print(f"     Actual:   {checkpoint['conversation_id']}")
        print(f"     Match: {'✓ YES' if test_id == checkpoint['conversation_id'] else '✗ NO'}")
        print()

    except Exception as e:
        print(f"  ❌ Error reading checkpoint: {e}")
        print()

    print()

print("=" * 80)
print()
print("💡 Tips for debugging:")
print("  1. Check the 'Chat Name' in checkpoint - does it match your upload?")
print("  2. Check the file count (total_files) - does it match your ZIP?")
print("  3. Look at the logs in your terminal when you upload the file")
print("  4. If name has (2), (3) suffix, that's from duplicate uploads")
print()
print("🔧 To force checkpoint detection:")
print("  - Upload the EXACT same ZIP file")
print("  - OR rename checkpoint file to match the new conversation ID")
print()
