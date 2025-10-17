#!/usr/bin/env python3
"""
Repair Checkpoint Encoding Script

This script fixes double-encoded UTF-8 text in checkpoint files.
It repairs transcriptions that have corrupted characters like '�' or 'Ã¨' instead of proper accents.

Usage:
    python repair_checkpoint_encoding.py [checkpoint_file]

If no file is specified, it will scan and repair all checkpoints in chat_checkpoints/
"""

import json
import os
import sys
import glob
from datetime import datetime


def fix_encoding(text: str) -> str:
    """
    Fix double-encoded UTF-8 text.

    Args:
        text: Potentially double-encoded text

    Returns:
        Correctly decoded text, or original if fix fails
    """
    if not text:
        return text

    # Check for Unicode replacement character (U+FFFD - �)
    replacement_char = '\ufffd'

    try:
        # Try to fix double-encoding: encode as latin-1, decode as utf-8
        fixed = text.encode('latin-1').decode('utf-8')

        # Only return fixed version if it looks better
        if replacement_char in text and replacement_char not in fixed:
            return fixed
        elif text != fixed:
            # Count "suspicious" characters in both
            suspicious_original = text.count(replacement_char) + text.count('Ã') + text.count('¨')
            suspicious_fixed = fixed.count(replacement_char) + fixed.count('Ã') + fixed.count('¨')

            if suspicious_fixed < suspicious_original:
                return fixed

        # No improvement, return original
        return text

    except (UnicodeDecodeError, UnicodeEncodeError, AttributeError):
        # If encoding fix fails, return original text
        return text


def repair_checkpoint(checkpoint_path: str, dry_run: bool = False) -> dict:
    """
    Repair encoding issues in a checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint JSON file
        dry_run: If True, only analyze without making changes

    Returns:
        Dictionary with repair statistics
    """
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Processing: {os.path.basename(checkpoint_path)}")
    print("-" * 80)

    try:
        # Load checkpoint
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)

        stats = {
            'total_transcriptions': len(checkpoint.get('transcriptions', {})),
            'fixed_count': 0,
            'fixed_corrected': 0,
            'corrupted_chars_before': 0,
            'corrupted_chars_after': 0
        }

        # Fix transcriptions
        replacement_char = '\ufffd'
        for filename, trans in checkpoint.get('transcriptions', {}).items():
            if 'text' in trans and trans['text']:
                original = trans['text']
                fixed = fix_encoding(original)

                if fixed != original:
                    stats['fixed_count'] += 1
                    stats['corrupted_chars_before'] += original.count(replacement_char)
                    stats['corrupted_chars_after'] += fixed.count(replacement_char)

                    print(f"\n  [OK] Fixed: {filename}")
                    # Show a sample of the change
                    if len(original) > 100:
                        print(f"    Before: {original[:100]}...")
                        print(f"    After:  {fixed[:100]}...")
                    else:
                        print(f"    Before: {original}")
                        print(f"    After:  {fixed}")

                    if not dry_run:
                        trans['text'] = fixed

            if 'corrected_text' in trans and trans['corrected_text']:
                original = trans['corrected_text']
                fixed = fix_encoding(original)

                if fixed != original:
                    stats['fixed_corrected'] += 1

                    if not dry_run:
                        trans['corrected_text'] = fixed

        # Save repaired checkpoint
        if not dry_run and (stats['fixed_count'] > 0 or stats['fixed_corrected'] > 0):
            # Create backup
            backup_path = checkpoint_path + '.backup'
            if not os.path.exists(backup_path):
                print(f"\n  Creating backup: {os.path.basename(backup_path)}")
                with open(backup_path, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint, f, indent=2, ensure_ascii=False)

            # Update timestamp
            checkpoint['timestamp'] = datetime.now().isoformat()

            # Save repaired checkpoint
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, indent=2, ensure_ascii=False)

            print(f"\n  [OK] Saved repaired checkpoint")

        # Print summary
        print(f"\n  Summary:")
        print(f"    Total transcriptions: {stats['total_transcriptions']}")
        print(f"    Fixed transcriptions: {stats['fixed_count']}")
        print(f"    Fixed corrected texts: {stats['fixed_corrected']}")
        print(f"    Corrupted chars removed: {stats['corrupted_chars_before'] - stats['corrupted_chars_after']}")

        return stats

    except Exception as e:
        print(f"  [ERROR] {e}")
        return None


def main():
    """Main script entry point."""
    print("=" * 80)
    print("CHECKPOINT ENCODING REPAIR TOOL")
    print("=" * 80)

    # Parse arguments
    dry_run = '--dry-run' in sys.argv or '--test' in sys.argv

    if dry_run:
        print("\n[!] DRY RUN MODE - No files will be modified")
        sys.argv = [arg for arg in sys.argv if arg not in ['--dry-run', '--test']]

    # Get checkpoint file(s)
    if len(sys.argv) > 1:
        # Specific file provided
        checkpoint_files = [sys.argv[1]]
    else:
        # Scan checkpoint directory
        checkpoint_dir = "chat_checkpoints"
        if not os.path.exists(checkpoint_dir):
            print(f"\n[ERROR] Checkpoint directory '{checkpoint_dir}' not found!")
            sys.exit(1)

        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*_checkpoint.json"))

        if not checkpoint_files:
            print(f"\n[ERROR] No checkpoint files found in '{checkpoint_dir}'")
            sys.exit(1)

        print(f"\nFound {len(checkpoint_files)} checkpoint file(s)")

    # Process each checkpoint
    total_stats = {
        'files_processed': 0,
        'files_fixed': 0,
        'total_fixed': 0
    }

    for checkpoint_path in checkpoint_files:
        if not os.path.exists(checkpoint_path):
            print(f"\n[ERROR] File not found: {checkpoint_path}")
            continue

        stats = repair_checkpoint(checkpoint_path, dry_run=dry_run)

        if stats:
            total_stats['files_processed'] += 1
            if stats['fixed_count'] > 0 or stats['fixed_corrected'] > 0:
                total_stats['files_fixed'] += 1
                total_stats['total_fixed'] += stats['fixed_count'] + stats['fixed_corrected']

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"  Files processed: {total_stats['files_processed']}")
    print(f"  Files with fixes: {total_stats['files_fixed']}")
    print(f"  Total items fixed: {total_stats['total_fixed']}")

    if dry_run:
        print("\n[!] This was a DRY RUN - no files were modified")
        print("  Run without --dry-run to apply fixes")
    elif total_stats['files_fixed'] > 0:
        print("\n[SUCCESS] Encoding issues have been repaired!")
        print("  Backups created with .backup extension")
        print("\n  Next steps:")
        print("  1. Restart your Streamlit app")
        print("  2. Upload the same ZIP file")
        print("  3. Click 'Resume Processing' (it will skip already-done files)")
        print("  4. Timeline will be regenerated with correct encoding")
    else:
        print("\n[OK] No encoding issues found - all checkpoints look good!")

    print()


if __name__ == "__main__":
    main()
