"""
Checkpoint Manager
Handles checkpoint creation, loading, and management for resilient processing.
"""

import json
import os
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages processing checkpoints for resilient transcription."""

    def __init__(self, checkpoint_dir: str = "chat_checkpoints"):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(f"CheckpointManager initialized with dir: {checkpoint_dir}")

    def generate_conversation_id(self, chat_name: str, date_range: str, file_count: int) -> str:
        """
        Generate unique conversation ID from chat metadata.

        Args:
            chat_name: Name of the conversation
            date_range: Date range string (e.g., "2024-01-01 to 2024-01-31")
            file_count: Number of audio files

        Returns:
            Unique conversation ID hash
        """
        unique_string = f"{chat_name}|{date_range}|{file_count}"
        conversation_id = hashlib.sha256(unique_string.encode()).hexdigest()[:16]
        logger.debug(f"Generated conversation ID: {conversation_id} for {chat_name}")
        return conversation_id

    def create_checkpoint(self, chat_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create new checkpoint for a conversation.

        Args:
            chat_data: Dictionary with chat metadata
                - conversation_id: Unique identifier
                - chat_name: Conversation name
                - total_files: Total number of audio files
                - date_range: Date range string (optional)

        Returns:
            New checkpoint data structure
        """
        checkpoint = {
            "conversation_id": chat_data['conversation_id'],
            "chat_name": chat_data['chat_name'],
            "timestamp": datetime.now().isoformat(),
            "total_files": chat_data['total_files'],
            "processed_files": 0,
            "failed_files": 0,
            "transcriptions": {},
            "processing_log": [],
            "stats": {
                "total_transcription_cost": 0.0,
                "total_llm_cost": 0.0,
                "total_duration_minutes": 0.0
            },
            "config": chat_data.get('config', {}),
            "date_range": chat_data.get('date_range', '')
        }

        logger.info(f"Created new checkpoint for {chat_data['chat_name']}")
        return checkpoint

    def save_checkpoint(self, checkpoint: Dict[str, Any], checkpoint_path: Optional[str] = None) -> str:
        """
        Save checkpoint to disk.

        Args:
            checkpoint: Checkpoint data structure
            checkpoint_path: Path to save to (optional, auto-generates if None)

        Returns:
            Path to saved checkpoint file
        """
        if checkpoint_path is None:
            conversation_id = checkpoint['conversation_id']
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"{conversation_id}_checkpoint.json"
            )

        # Update timestamp
        checkpoint['timestamp'] = datetime.now().isoformat()

        try:
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, indent=2, ensure_ascii=False)

            logger.debug(f"Saved checkpoint to {checkpoint_path}")
            return checkpoint_path

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load checkpoint from disk.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Checkpoint data structure

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            json.JSONDecodeError: If checkpoint is corrupted
        """
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)

            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            logger.info(f"  - Processed: {checkpoint['processed_files']}/{checkpoint['total_files']}")
            logger.info(f"  - Failed: {checkpoint['failed_files']}")

            return checkpoint

        except FileNotFoundError:
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Corrupted checkpoint file: {checkpoint_path} - {e}")
            raise

    def find_checkpoint_for_chat(self, conversation_id: str) -> Optional[str]:
        """
        Find existing checkpoint for a conversation.

        Args:
            conversation_id: Unique conversation identifier

        Returns:
            Path to checkpoint file if found, None otherwise
        """
        checkpoint_filename = f"{conversation_id}_checkpoint.json"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_filename)

        if os.path.exists(checkpoint_path):
            logger.info(f"Found existing checkpoint: {checkpoint_path}")
            return checkpoint_path

        logger.debug(f"No checkpoint found for conversation {conversation_id}")
        return None

    def get_pending_files(self, checkpoint: Dict[str, Any], all_files: List[str]) -> List[str]:
        """
        Get list of files not yet processed.

        Args:
            checkpoint: Checkpoint data structure
            all_files: List of all audio file paths

        Returns:
            List of file paths not yet processed
        """
        processed_files = set(checkpoint['transcriptions'].keys())

        # Convert paths to just filenames for comparison
        pending = []
        for file_path in all_files:
            filename = os.path.basename(file_path)
            if filename not in processed_files:
                pending.append(file_path)

        logger.info(f"Found {len(pending)} pending files out of {len(all_files)} total")
        return pending

    def get_failed_files(self, checkpoint: Dict[str, Any], all_files: List[str]) -> List[str]:
        """
        Get list of files that failed transcription.

        Args:
            checkpoint: Checkpoint data structure
            all_files: List of all audio file paths

        Returns:
            List of file paths that failed
        """
        failed_filenames = set()
        for filename, trans_record in checkpoint['transcriptions'].items():
            if not trans_record.get('success'):
                failed_filenames.add(filename)

        # Convert back to full paths
        failed = []
        for file_path in all_files:
            filename = os.path.basename(file_path)
            if filename in failed_filenames:
                failed.append(file_path)

        logger.info(f"Found {len(failed)} failed files out of {len(all_files)} total")
        return failed

    def add_transcription_result(
        self,
        checkpoint: Dict[str, Any],
        filename: str,
        result: Dict[str, Any],
        file_info: Optional[Dict[str, Any]] = None
    ):
        """
        Add transcription result to checkpoint.

        Args:
            checkpoint: Checkpoint data structure
            filename: Audio filename
            result: Transcription result dictionary
            file_info: Optional file metadata (size, duration, etc.)
        """
        # Store just the filename, not full path
        filename_only = os.path.basename(filename)

        # Build transcription record
        transcription_record = {
            "success": result.get('success', False),
            "timestamp": datetime.now().isoformat()
        }

        if result.get('success'):
            transcription_record.update({
                "text": result.get('text', ''),
                "language": result.get('language'),
                "duration": result.get('duration', 0.0),
                "cost": result.get('cost', 0.0)
            })

            # Add corrected text if available
            if 'corrected_text' in result:
                transcription_record['corrected_text'] = result['corrected_text']

        else:
            transcription_record.update({
                "error": result.get('error', 'Unknown error'),
                "attempts": result.get('attempts', 1)
            })
            checkpoint['failed_files'] += 1

        # Add file info if provided
        if file_info:
            transcription_record.update({
                "file_size_bytes": file_info.get('size_bytes'),
                "audio_duration_seconds": file_info.get('duration_seconds')
            })

        # Store in checkpoint
        checkpoint['transcriptions'][filename_only] = transcription_record
        checkpoint['processed_files'] = len(checkpoint['transcriptions'])

        # Update stats
        if result.get('success'):
            cost = result.get('cost', 0.0)
            checkpoint['stats']['total_transcription_cost'] += cost

            duration = result.get('duration', 0.0)
            checkpoint['stats']['total_duration_minutes'] += duration / 60.0

        logger.debug(f"Added transcription result for {filename_only}: success={result.get('success')}")

    def add_log_entry(
        self,
        checkpoint: Dict[str, Any],
        event_type: str,
        filename: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Add entry to processing log.

        Args:
            checkpoint: Checkpoint data structure
            event_type: Type of event (e.g., 'started', 'transcription_complete', 'timeout')
            filename: Audio filename
            details: Optional additional details
        """
        filename_only = os.path.basename(filename)

        log_entry = {
            "time": datetime.now().isoformat(),
            "event": event_type,
            "file": filename_only
        }

        if details:
            log_entry.update(details)

        checkpoint['processing_log'].append(log_entry)

        logger.debug(f"Added log entry: {event_type} for {filename_only}")

    def get_failed_files(self, checkpoint: Dict[str, Any]) -> List[str]:
        """
        Get list of failed file names.

        Args:
            checkpoint: Checkpoint data structure

        Returns:
            List of filenames that failed
        """
        failed = [
            filename for filename, data in checkpoint['transcriptions'].items()
            if not data.get('success', False)
        ]

        logger.info(f"Found {len(failed)} failed files")
        return failed

    def update_llm_cost(self, checkpoint: Dict[str, Any], cost: float):
        """
        Update LLM correction cost in checkpoint.

        Args:
            checkpoint: Checkpoint data structure
            cost: LLM correction cost to add
        """
        checkpoint['stats']['total_llm_cost'] += cost
        logger.debug(f"Updated LLM cost: +${cost:.4f}, total: ${checkpoint['stats']['total_llm_cost']:.4f}")

    def cleanup_old_checkpoints(self, days: int = 7):
        """
        Remove checkpoint files older than specified days.

        Args:
            days: Maximum age of checkpoints to keep
        """
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            removed_count = 0

            for filename in os.listdir(self.checkpoint_dir):
                if not filename.endswith('_checkpoint.json'):
                    continue

                filepath = os.path.join(self.checkpoint_dir, filename)

                # Check file modification time
                mtime = datetime.fromtimestamp(os.path.getmtime(filepath))

                if mtime < cutoff_time:
                    os.remove(filepath)
                    removed_count += 1
                    logger.info(f"Removed old checkpoint: {filename}")

            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old checkpoint(s)")
            else:
                logger.debug("No old checkpoints to clean up")

        except Exception as e:
            logger.error(f"Error during checkpoint cleanup: {e}")

    def export_processing_log(self, checkpoint: Dict[str, Any], output_path: str):
        """
        Export processing log to CSV for analysis.

        Args:
            checkpoint: Checkpoint data structure
            output_path: Path to save CSV file
        """
        import pandas as pd

        try:
            df = pd.DataFrame(checkpoint['processing_log'])

            if not df.empty:
                df.to_csv(output_path, index=False)
                logger.info(f"Exported processing log to {output_path}")
            else:
                logger.warning("Processing log is empty, nothing to export")

        except Exception as e:
            logger.error(f"Failed to export processing log: {e}")
            raise

    def get_checkpoint_summary(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get summary statistics from checkpoint.

        Args:
            checkpoint: Checkpoint data structure

        Returns:
            Summary dictionary with key stats
        """
        total = checkpoint['total_files']
        processed = checkpoint['processed_files']
        failed = checkpoint['failed_files']
        succeeded = processed - failed

        summary = {
            "conversation_id": checkpoint['conversation_id'],
            "chat_name": checkpoint['chat_name'],
            "last_updated": checkpoint['timestamp'],
            "progress": {
                "total_files": total,
                "processed_files": processed,
                "succeeded_files": succeeded,
                "failed_files": failed,
                "progress_percent": (processed / total * 100) if total > 0 else 0,
                "remaining_files": total - processed
            },
            "costs": {
                "transcription_cost": checkpoint['stats']['total_transcription_cost'],
                "llm_cost": checkpoint['stats']['total_llm_cost'],
                "total_cost": checkpoint['stats']['total_transcription_cost'] + checkpoint['stats']['total_llm_cost']
            },
            "duration": {
                "total_minutes": checkpoint['stats']['total_duration_minutes'],
                "average_seconds_per_file": (checkpoint['stats']['total_duration_minutes'] * 60 / processed) if processed > 0 else 0
            }
        }

        return summary

    def delete_checkpoint(self, checkpoint_path: str):
        """
        Delete a checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint file to delete
        """
        try:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                logger.info(f"Deleted checkpoint: {checkpoint_path}")
            else:
                logger.warning(f"Checkpoint not found for deletion: {checkpoint_path}")

        except Exception as e:
            logger.error(f"Failed to delete checkpoint: {e}")
            raise

    def export_processing_log(self, checkpoint: Dict[str, Any]) -> str:
        """
        Export processing log as CSV.

        Args:
            checkpoint: Checkpoint data structure

        Returns:
            CSV content as string
        """
        import io
        import csv

        output = io.StringIO()
        writer = csv.writer(output)

        # Write summary header
        output.write("# WhatsApp Transcriber Processing Log\n")
        output.write(f"# Chat: {checkpoint['chat_name']}\n")
        output.write(f"# Conversation ID: {checkpoint['conversation_id']}\n")
        output.write(f"# Generated: {datetime.now().isoformat()}\n")
        output.write(f"# Last Updated: {checkpoint['timestamp']}\n")
        output.write("#\n")
        output.write(f"# Total Files: {checkpoint['total_files']}\n")
        output.write(f"# Processed: {checkpoint['processed_files']}\n")
        output.write(f"# Failed: {checkpoint['failed_files']}\n")
        output.write(f"# Success Rate: {((checkpoint['processed_files'] - checkpoint['failed_files']) / checkpoint['total_files'] * 100):.1f}%\n")
        output.write(f"# Total Cost: ${checkpoint['stats']['total_transcription_cost'] + checkpoint['stats']['total_llm_cost']:.4f}\n")
        output.write("#\n")

        # Write event log
        output.write("# Processing Event Log\n")
        writer.writerow(['Timestamp', 'Event Type', 'File', 'Duration (s)', 'Text Length', 'Error', 'Details'])

        for log_entry in checkpoint.get('processing_log', []):
            writer.writerow([
                log_entry.get('time', ''),
                log_entry.get('event', ''),
                log_entry.get('file', ''),
                log_entry.get('duration', ''),
                log_entry.get('text_length', ''),
                log_entry.get('error', ''),
                json.dumps(log_entry.get('details', {})) if log_entry.get('details') else ''
            ])

        output.write("\n")

        # Write transcription results summary
        output.write("# File Results Summary\n")
        writer.writerow(['Filename', 'Status', 'Text Length', 'Language', 'Duration (s)', 'Cost', 'Error', 'Timestamp'])

        for filename, trans_record in checkpoint.get('transcriptions', {}).items():
            writer.writerow([
                filename,
                'Success' if trans_record.get('success') else 'Failed',
                trans_record.get('text_length', len(trans_record.get('text', ''))),
                trans_record.get('language', ''),
                trans_record.get('duration', ''),
                trans_record.get('cost', ''),
                trans_record.get('error', ''),
                trans_record.get('timestamp', '')
            ])

        logger.info(f"Exported processing log with {len(checkpoint.get('processing_log', []))} events")
        return output.getvalue()


# Convenience functions for simple usage
def create_checkpoint(chat_data: Dict[str, Any], checkpoint_dir: str = "chat_checkpoints") -> Dict[str, Any]:
    """Create a new checkpoint."""
    manager = CheckpointManager(checkpoint_dir)
    return manager.create_checkpoint(chat_data)


def save_checkpoint(checkpoint: Dict[str, Any], checkpoint_path: Optional[str] = None, checkpoint_dir: str = "chat_checkpoints") -> str:
    """Save checkpoint to disk."""
    manager = CheckpointManager(checkpoint_dir)
    return manager.save_checkpoint(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load checkpoint from disk."""
    manager = CheckpointManager()
    return manager.load_checkpoint(checkpoint_path)


def find_checkpoint_for_chat(conversation_id: str, checkpoint_dir: str = "chat_checkpoints") -> Optional[str]:
    """Find existing checkpoint for a conversation."""
    manager = CheckpointManager(checkpoint_dir)
    return manager.find_checkpoint_for_chat(conversation_id)


def get_pending_files(checkpoint: Dict[str, Any], all_files: List[str]) -> List[str]:
    """Get list of unprocessed files."""
    manager = CheckpointManager()
    return manager.get_pending_files(checkpoint, all_files)
