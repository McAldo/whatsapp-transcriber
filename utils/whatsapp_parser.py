"""
WhatsApp Chat Parser
Handles parsing of WhatsApp chat export files in various formats.
"""

import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class WhatsAppMessage:
    """Represents a single WhatsApp message."""

    def __init__(self, timestamp: Optional[datetime], sender: str, content: str,
                 message_type: str = "text", media_file: Optional[str] = None):
        self.timestamp = timestamp
        self.sender = sender
        self.content = content
        self.message_type = message_type  # text, voice, image, video, document, etc.
        self.media_file = media_file

    def __repr__(self):
        return f"WhatsAppMessage({self.timestamp}, {self.sender}, {self.message_type})"


class WhatsAppParser:
    """Parser for WhatsApp chat export files."""

    # Common WhatsApp timestamp patterns
    PATTERNS = [
        # Pattern 1: [DD/MM/YYYY, HH:MM:SS] or [DD/MM/YY, HH:MM:SS]
        r'\[(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?)\]\s*([^:]+?):\s*(.*)',
        # Pattern 2: DD/MM/YYYY, HH:MM - or DD/MM/YY, HH:MM -
        r'(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?)\s*-\s*([^:]+?):\s*(.*)',
        # Pattern 3: DD.MM.YY, HH:MM - (German format)
        r'(\d{1,2}\.\d{1,2}\.\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?)\s*-\s*([^:]+?):\s*(.*)',
        # Pattern 4: M/D/YY, H:MM AM/PM - (US format)
        r'(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?\s?[AP]M)\s*-\s*([^:]+?):\s*(.*)',
    ]

    # Media attachment patterns
    MEDIA_PATTERNS = {
        'image': r'<attached:\s*([^>]+\.(?:jpg|jpeg|png|gif|webp))>|(.+\.(?:jpg|jpeg|png|gif|webp))\s*\(file attached\)',
        'video': r'<attached:\s*([^>]+\.(?:mp4|avi|mov|3gp))>|(.+\.(?:mp4|avi|mov|3gp))\s*\(file attached\)',
        'audio': r'<attached:\s*([^>]+\.(?:opus|mp3|m4a|aac))>|(.+\.(?:opus|mp3|m4a|aac))\s*\(file attached\)|PTT-\d+-WA\d+\.opus',
        'document': r'<attached:\s*([^>]+\.(?:pdf|doc|docx|txt|xlsx))>|(.+\.(?:pdf|doc|docx|txt|xlsx))\s*\(file attached\)',
    }

    def __init__(self):
        self.messages: List[WhatsAppMessage] = []
        self.current_message: Optional[WhatsAppMessage] = None

    def parse_file(self, file_path: str) -> List[WhatsAppMessage]:
        """
        Parse a WhatsApp chat export text file.

        Args:
            file_path: Path to the chat text file

        Returns:
            List of WhatsAppMessage objects
        """
        self.messages = []
        self.current_message = None

        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            content = None

            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.readlines()
                    logger.info(f"Successfully read file with {encoding} encoding")
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue

            if content is None:
                raise ValueError("Could not read file with any supported encoding")

            for line in content:
                line = line.rstrip('\n\r')
                if not line.strip():
                    continue

                # Try to parse as a new message
                parsed = self._parse_line(line)

                if parsed:
                    # Save previous message if exists
                    if self.current_message:
                        self.messages.append(self.current_message)

                    # Start new message
                    timestamp, sender, content = parsed
                    message_type, media_file = self._detect_media(content)
                    self.current_message = WhatsAppMessage(
                        timestamp, sender, content, message_type, media_file
                    )
                else:
                    # Continuation of previous message
                    if self.current_message:
                        self.current_message.content += '\n' + line

            # Don't forget the last message
            if self.current_message:
                self.messages.append(self.current_message)

            logger.info(f"Parsed {len(self.messages)} messages")
            return self.messages

        except Exception as e:
            logger.error(f"Error parsing WhatsApp file: {e}")
            raise

    def _parse_line(self, line: str) -> Optional[Tuple[datetime, str, str]]:
        """
        Try to parse a line as a WhatsApp message.

        Returns:
            Tuple of (timestamp, sender, content) or None if not a valid message line
        """
        for pattern in self.PATTERNS:
            match = re.match(pattern, line)
            if match:
                try:
                    date_str = match.group(1)
                    time_str = match.group(2)
                    sender = match.group(3).strip()
                    content = match.group(4).strip()

                    # Parse timestamp
                    timestamp = self._parse_timestamp(date_str, time_str)

                    return timestamp, sender, content
                except Exception as e:
                    logger.debug(f"Failed to parse matched line: {e}")
                    continue

        return None

    def _parse_timestamp(self, date_str: str, time_str: str) -> datetime:
        """
        Parse WhatsApp timestamp from date and time strings.
        Handles multiple formats.
        """
        # Normalize the strings
        date_str = date_str.strip()
        time_str = time_str.strip()

        # Try different date formats
        date_formats = [
            '%d/%m/%Y', '%d/%m/%y',  # DD/MM/YYYY or DD/MM/YY
            '%m/%d/%Y', '%m/%d/%y',  # MM/DD/YYYY or MM/DD/YY (US format)
            '%d.%m.%Y', '%d.%m.%y',  # DD.MM.YYYY or DD.MM.YY (German format)
        ]

        # Try different time formats
        time_formats = [
            '%H:%M:%S',      # 24-hour with seconds
            '%H:%M',         # 24-hour without seconds
            '%I:%M:%S %p',   # 12-hour with seconds
            '%I:%M %p',      # 12-hour without seconds
            '%I:%M:%S%p',    # 12-hour with seconds (no space)
            '%I:%M%p',       # 12-hour without seconds (no space)
        ]

        # Try all combinations
        for date_fmt in date_formats:
            for time_fmt in time_formats:
                try:
                    combined = f"{date_str} {time_str}"
                    timestamp = datetime.strptime(combined, f"{date_fmt} {time_fmt}")
                    return timestamp
                except ValueError:
                    continue

        # If all else fails, try to at least get the date
        for date_fmt in date_formats:
            try:
                timestamp = datetime.strptime(date_str, date_fmt)
                logger.warning(f"Could not parse time '{time_str}', using date only")
                return timestamp
            except ValueError:
                continue

        raise ValueError(f"Could not parse timestamp: {date_str} {time_str}")

    def _detect_media(self, content: str) -> Tuple[str, Optional[str]]:
        """
        Detect if message contains media attachment.

        Returns:
            Tuple of (message_type, media_filename)
        """
        # Check for media patterns
        for media_type, pattern in self.MEDIA_PATTERNS.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                # Extract filename from any capture group that matched
                filename = next((g for g in match.groups() if g), None)

                # Special handling for voice messages
                if media_type == 'audio' and ('PTT-' in content or 'voice message' in content.lower()):
                    return 'voice', filename

                return media_type, filename

        # Check for common media indicators
        lower_content = content.lower()
        if any(x in lower_content for x in ['image omitted', 'photo omitted', '📷']):
            return 'image', None
        if any(x in lower_content for x in ['video omitted', '🎥', '📹']):
            return 'video', None
        if any(x in lower_content for x in ['audio omitted', 'voice message', '🎤']):
            return 'voice', None
        if any(x in lower_content for x in ['document omitted', '📄']):
            return 'document', None
        if any(x in lower_content for x in ['sticker omitted', 'gif omitted']):
            return 'sticker', None

        return 'text', None

    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about parsed messages."""
        stats = {
            'total': len(self.messages),
            'text': 0,
            'voice': 0,
            'image': 0,
            'video': 0,
            'document': 0,
            'other': 0,
        }

        for msg in self.messages:
            if msg.message_type in stats:
                stats[msg.message_type] += 1
            else:
                stats['other'] += 1

        return stats
