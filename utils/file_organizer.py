"""
File Organizer
Handles extraction, organization, and management of WhatsApp media files.
"""

import os
import shutil
import zipfile
import tempfile
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import re

logger = logging.getLogger(__name__)


class FileOrganizer:
    """Handles file extraction and organization."""

    MEDIA_EXTENSIONS = {
        'image': ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'],
        'video': ['.mp4', '.avi', '.mov', '.3gp', '.mkv', '.webm'],
        'audio': ['.opus', '.mp3', '.m4a', '.aac', '.ogg', '.wav'],
        'document': ['.pdf', '.doc', '.docx', '.txt', '.xlsx', '.xls', '.ppt', '.pptx'],
    }

    CHAT_FILE_PATTERNS = [
        '_chat.txt',
        'WhatsApp Chat with',
        '.txt'
    ]

    def __init__(self):
        self.temp_dir = None
        self.extracted_files = []
        self.media_files = {}
        self.chat_file = None

    def extract_zip(self, zip_path: str) -> str:
        """
        Extract WhatsApp export ZIP file to temporary directory.

        Args:
            zip_path: Path to ZIP file

        Returns:
            Path to extraction directory
        """
        try:
            # Create temporary directory
            self.temp_dir = tempfile.mkdtemp(prefix='whatsapp_export_')
            logger.info(f"Created temp directory: {self.temp_dir}")

            # Extract ZIP
            logger.info(f"Extracting ZIP: {zip_path}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.temp_dir)

            # List all extracted files
            self._scan_extracted_files()

            logger.info(f"Extracted {len(self.extracted_files)} files")
            return self.temp_dir

        except zipfile.BadZipFile:
            raise ValueError("Invalid ZIP file. Please ensure you uploaded a WhatsApp export ZIP.")
        except Exception as e:
            logger.error(f"Error extracting ZIP: {e}")
            raise

    def _scan_extracted_files(self):
        """Scan extracted directory and categorize files."""
        self.extracted_files = []
        self.media_files = {
            'image': [],
            'video': [],
            'audio': [],
            'document': [],
            'other': []
        }
        self.chat_file = None

        for root, dirs, files in os.walk(self.temp_dir):
            for file in files:
                filepath = os.path.join(root, file)
                self.extracted_files.append(filepath)

                # Check if it's the chat file
                if self._is_chat_file(file):
                    self.chat_file = filepath
                    logger.info(f"Found chat file: {file}")
                    continue

                # Categorize media files
                ext = os.path.splitext(file)[1].lower()
                categorized = False

                for media_type, extensions in self.MEDIA_EXTENSIONS.items():
                    if ext in extensions:
                        self.media_files[media_type].append(filepath)
                        categorized = True
                        break

                if not categorized and ext:  # Has extension but not in known types
                    self.media_files['other'].append(filepath)

        # Log statistics
        total_media = sum(len(files) for files in self.media_files.values())
        logger.info(f"Found {total_media} media files:")
        for media_type, files in self.media_files.items():
            if files:
                logger.info(f"  {media_type}: {len(files)}")

    def _is_chat_file(self, filename: str) -> bool:
        """Check if file is the WhatsApp chat text file."""
        for pattern in self.CHAT_FILE_PATTERNS:
            if pattern in filename:
                return True
        return False

    def get_chat_file(self) -> Optional[str]:
        """Get path to the chat text file."""
        return self.chat_file

    def get_audio_files(self) -> List[str]:
        """Get list of audio files (for transcription)."""
        return self.media_files.get('audio', [])

    def get_all_media_files(self) -> List[str]:
        """Get list of all media files."""
        all_media = []
        for files in self.media_files.values():
            all_media.extend(files)
        return all_media

    def organize_media(self, output_dir: str, timestamps: Optional[Dict[str, datetime]] = None) -> Dict[str, str]:
        """
        Copy and organize media files to output directory with proper naming.

        Args:
            output_dir: Directory to copy files to
            timestamps: Optional dictionary mapping original filenames to timestamps

        Returns:
            Dictionary mapping original paths to new paths
        """
        os.makedirs(output_dir, exist_ok=True)
        file_mapping = {}

        if timestamps is None:
            timestamps = {}

        for filepath in self.get_all_media_files():
            try:
                original_filename = os.path.basename(filepath)

                # Get timestamp for this file
                timestamp = timestamps.get(original_filename)

                if timestamp is None:
                    # Try to extract from filename
                    timestamp = self._extract_timestamp_from_filename(original_filename)

                if timestamp is None:
                    # Use file modification time
                    timestamp = datetime.fromtimestamp(os.path.getmtime(filepath))

                # Create new filename with timestamp prefix
                new_filename = self._create_timestamped_filename(original_filename, timestamp)

                # Copy file
                new_path = os.path.join(output_dir, new_filename)
                shutil.copy2(filepath, new_path)

                file_mapping[filepath] = new_path
                logger.debug(f"Organized: {original_filename} -> {new_filename}")

            except Exception as e:
                logger.error(f"Error organizing file {filepath}: {e}")
                continue

        logger.info(f"Organized {len(file_mapping)} media files to {output_dir}")
        return file_mapping

    def _extract_timestamp_from_filename(self, filename: str) -> Optional[datetime]:
        """
        Try to extract timestamp from WhatsApp filename patterns.

        Examples:
        - PTT-20231015-WA0001.opus (voice messages)
        - IMG-20231015-WA0001.jpg (images)
        - VID-20231015-WA0001.mp4 (videos)
        """
        # Pattern: XXX-YYYYMMDD-WA####.ext
        match = re.search(r'(?:PTT|IMG|VID|AUD|DOC)-(\d{8})-WA\d+', filename)
        if match:
            date_str = match.group(1)
            try:
                return datetime.strptime(date_str, '%Y%m%d')
            except ValueError:
                pass

        return None

    def _create_timestamped_filename(self, original_filename: str, timestamp: datetime) -> str:
        """
        Create filename with timestamp prefix.

        Format: YYYYMMDD_HHMMSS_original_filename
        """
        # Format timestamp
        timestamp_str = timestamp.strftime('%Y%m%d_%H%M%S')

        # Remove any existing timestamp prefix from original
        clean_name = re.sub(r'^\d{8}_\d{6}_', '', original_filename)

        # Combine
        new_filename = f"{timestamp_str}_{clean_name}"

        return new_filename

    def create_media_index(self, file_mapping: Dict[str, str], output_path: str):
        """
        Create an index file listing all media files.

        Args:
            file_mapping: Dictionary mapping original paths to new paths
            output_path: Path to write index file
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("WhatsApp Media Files Index\n")
                f.write("=" * 80 + "\n\n")

                for original_path, new_path in sorted(file_mapping.items()):
                    original_name = os.path.basename(original_path)
                    new_name = os.path.basename(new_path)

                    # Get file size
                    try:
                        size = os.path.getsize(new_path)
                        size_str = self._format_file_size(size)
                    except:
                        size_str = "Unknown"

                    f.write(f"Original: {original_name}\n")
                    f.write(f"New Name: {new_name}\n")
                    f.write(f"Size: {size_str}\n")
                    f.write("-" * 80 + "\n")

            logger.info(f"Created media index: {output_path}")

        except Exception as e:
            logger.error(f"Error creating media index: {e}")

    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

    def get_statistics(self) -> Dict[str, any]:
        """Get statistics about extracted files."""
        stats = {
            'total_files': len(self.extracted_files),
            'chat_file_found': self.chat_file is not None,
            'media_files': {}
        }

        for media_type, files in self.media_files.items():
            stats['media_files'][media_type] = len(files)

        return stats

    def cleanup(self):
        """Clean up temporary directory."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temp directory: {self.temp_dir}")
            except Exception as e:
                logger.error(f"Error cleaning up temp directory: {e}")


def create_output_zip(source_dir: str, output_zip_path: str) -> str:
    """
    Create a ZIP file from directory contents.

    Args:
        source_dir: Directory to zip
        output_zip_path: Path for output ZIP file

    Returns:
        Path to created ZIP file
    """
    try:
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, source_dir)
                    zipf.write(file_path, arcname)

        logger.info(f"Created ZIP: {output_zip_path}")
        return output_zip_path

    except Exception as e:
        logger.error(f"Error creating ZIP: {e}")
        raise
