"""
Voxtral Transcriber
Handles transcription of audio files using Mistral's Voxtral Mini API.
"""

import os
import logging
from typing import Optional, Dict, List
from pathlib import Path

logger = logging.getLogger(__name__)


class VoxtralTranscriber:
    """Handles audio transcription using Mistral's Voxtral Mini API."""

    def __init__(self, api_key: str):
        """
        Initialize the Voxtral transcriber.

        Args:
            api_key: Mistral API key
        """
        self.api_key = api_key
        self.client = None
        self.total_duration = 0.0  # Track total audio duration for cost calculation
        logger.info("VoxtralTranscriber initialized")

    def initialize(self):
        """Initialize the Mistral client."""
        try:
            from mistralai import Mistral
            self.client = Mistral(api_key=self.api_key)
            logger.info("Mistral client initialized successfully")
        except ImportError:
            raise ImportError("mistralai package not found. Install with: pip install mistralai>=1.0.0")
        except Exception as e:
            logger.error(f"Failed to initialize Mistral client: {e}")
            raise

    def test_connection(self) -> Dict[str, any]:
        """
        Test the Mistral API connection.

        Returns:
            Dictionary with connection test results
        """
        try:
            if self.client is None:
                self.initialize()

            # Try to list models to verify API key
            models = self.client.models.list()

            return {
                'success': True,
                'message': 'Connection successful',
                'models_available': len(models.data) if hasattr(models, 'data') else 0
            }
        except Exception as e:
            error_msg = str(e).lower()
            if 'unauthorized' in error_msg or 'api key' in error_msg:
                return {
                    'success': False,
                    'message': 'Invalid API key',
                    'error': str(e)
                }
            else:
                return {
                    'success': False,
                    'message': 'Connection failed',
                    'error': str(e)
                }

    def _fix_encoding(self, text: str) -> str:
        """
        Fix double-encoded UTF-8 text.

        Some APIs return UTF-8 text that was incorrectly decoded as Latin-1/ISO-8859-1,
        causing characters like 'è' to appear as '�' or '\u00c3\u00a8'.

        This function attempts to fix such encoding issues by:
        1. Encoding the text back to Latin-1 bytes
        2. Decoding those bytes as UTF-8

        Args:
            text: Potentially double-encoded text

        Returns:
            Correctly decoded text
        """
        if not text:
            return text

        try:
            # Try to fix double-encoding: encode as latin-1, decode as utf-8
            fixed = text.encode('latin-1').decode('utf-8')

            # Only return fixed version if it looks better (no replacement chars)
            if '�' in text and '�' not in fixed:
                logger.debug(f"Fixed encoding issue in text (before: {len([c for c in text if c == '�'])} replacement chars)")
                return fixed
            elif text != fixed:
                # Text changed, but let's verify it's actually better
                # Count "suspicious" characters in both
                suspicious_original = text.count('�') + text.count('Ã') + text.count('¨')
                suspicious_fixed = fixed.count('�') + fixed.count('Ã') + fixed.count('¨')

                if suspicious_fixed < suspicious_original:
                    logger.debug("Applied encoding fix")
                    return fixed

            # No improvement, return original
            return text

        except (UnicodeDecodeError, UnicodeEncodeError, AttributeError):
            # If encoding fix fails, return original text
            logger.debug("Could not apply encoding fix, using original")
            return text

    def transcribe_file(
        self,
        audio_path: str,
        language: Optional[str] = None,
        max_retries: int = 3
    ) -> Dict[str, any]:
        """
        Transcribe a single audio file using Voxtral Mini.

        Args:
            audio_path: Path to audio file
            language: Optional language code (e.g., 'en', 'it'). Auto-detect if None.
            max_retries: Maximum number of retry attempts for failures

        Returns:
            Dictionary with transcription results
        """
        if self.client is None:
            self.initialize()

        # Validate file format
        supported_formats = ['.mp3', '.wav', '.flac', '.ogg', '.opus']
        file_ext = Path(audio_path).suffix.lower()

        if file_ext not in supported_formats:
            logger.warning(f"Unsupported audio format: {file_ext}")
            return {
                'text': '',
                'language': None,
                'duration': 0.0,
                'success': False,
                'error': f'Unsupported audio format: {file_ext}. Supported formats: {", ".join(supported_formats)}'
            }

        # Try transcription with retries
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Transcribing with Voxtral (attempt {attempt}/{max_retries}): {audio_path}")

                with open(audio_path, "rb") as audio_file:
                    # Upload the file
                    uploaded = self.client.files.upload(
                        file={
                            "content": audio_file,
                            "file_name": os.path.basename(audio_path)
                        },
                        purpose="audio"
                    )

                    logger.debug(f"File uploaded successfully. File ID: {uploaded.id}")

                    # Get signed URL
                    signed_url = self.client.files.get_signed_url(file_id=uploaded.id)

                    logger.debug(f"Got signed URL for file")

                    # Prepare transcription parameters
                    transcribe_params = {
                        "model": "voxtral-mini-latest",
                        "file_url": signed_url.url
                    }

                    # Add language parameter if specified (not for auto-detect)
                    if language and language != 'auto':
                        transcribe_params["language"] = language
                        logger.debug(f"Using language: {language}")

                    # Transcribe
                    transcription = self.client.audio.transcriptions.complete(**transcribe_params)

                    # Get duration if available (for cost calculation)
                    duration = getattr(transcription, 'duration', 0.0)
                    if duration:
                        self.total_duration += duration

                    # Get detected language if available
                    detected_lang = getattr(transcription, 'language', None)

                    # Fix encoding issues (UTF-8 double-encoding)
                    fixed_text = self._fix_encoding(transcription.text)

                    result = {
                        'text': fixed_text,
                        'language': detected_lang,
                        'duration': duration,
                        'success': True,
                        'error': None
                    }

                    logger.info(f"Transcription complete: {len(transcription.text)} characters")
                    if detected_lang:
                        logger.info(f"Detected language: {detected_lang}")

                    return result

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Voxtral transcription failed (attempt {attempt}/{max_retries}): {error_msg}")

                # Check for specific error types
                if 'rate limit' in error_msg.lower():
                    if attempt < max_retries:
                        import time
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.info(f"Rate limited. Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return {
                            'text': '',
                            'language': None,
                            'duration': 0.0,
                            'success': False,
                            'error': 'Rate limit exceeded. Please try again later.'
                        }

                elif 'unauthorized' in error_msg.lower() or 'api key' in error_msg.lower():
                    return {
                        'text': '',
                        'language': None,
                        'duration': 0.0,
                        'success': False,
                        'error': 'Invalid API key. Please check your Mistral API key.'
                    }

                elif attempt < max_retries:
                    # Generic error, retry
                    logger.info(f"Retrying transcription...")
                    continue
                else:
                    # Max retries reached
                    return {
                        'text': '',
                        'language': None,
                        'duration': 0.0,
                        'success': False,
                        'error': f'Transcription failed after {max_retries} attempts: {error_msg}'
                    }

        # Should not reach here, but just in case
        return {
            'text': '',
            'language': None,
            'duration': 0.0,
            'success': False,
            'error': 'Unknown error occurred'
        }

    def transcribe_batch(
        self,
        audio_files: List[str],
        language: Optional[str] = None
    ) -> Dict[str, Dict]:
        """
        Transcribe multiple audio files.

        Args:
            audio_files: List of audio file paths
            language: Optional language code

        Returns:
            Dictionary mapping filenames to transcription results
        """
        results = {}
        total = len(audio_files)

        for idx, audio_file in enumerate(audio_files, 1):
            filename = os.path.basename(audio_file)
            logger.info(f"Processing file {idx}/{total}: {filename}")

            result = self.transcribe_file(audio_file, language)
            results[audio_file] = result

        return results

    def get_estimated_cost(self, duration_seconds: float) -> float:
        """
        Calculate estimated cost for audio transcription.

        Args:
            duration_seconds: Audio duration in seconds

        Returns:
            Estimated cost in USD
        """
        # Voxtral pricing: $0.001 per minute
        duration_minutes = duration_seconds / 60.0
        cost = duration_minutes * 0.001
        return cost

    def get_total_cost(self) -> float:
        """
        Get total cost for all transcriptions processed.

        Returns:
            Total cost in USD
        """
        return self.get_estimated_cost(self.total_duration)

    def reset_cost_tracking(self):
        """Reset the total duration counter for cost tracking."""
        self.total_duration = 0.0


def get_audio_duration(audio_path: str) -> float:
    """
    Get duration of an audio file.

    Args:
        audio_path: Path to audio file

    Returns:
        Duration in seconds, or 0.0 if unable to determine
    """
    try:
        # Try using faster-whisper's built-in audio loading
        from faster_whisper.audio import decode_audio
        import numpy as np

        audio = decode_audio(audio_path)
        # Audio is sampled at 16kHz by default in faster-whisper
        duration = len(audio) / 16000.0
        return duration
    except Exception as e:
        logger.debug(f"Could not determine duration for {audio_path}: {e}")
        # Fallback: estimate based on file size (very rough estimate)
        # Average bitrate for opus: ~20-40 kbps, we'll use 30
        try:
            file_size = os.path.getsize(audio_path)
            # Rough estimate: file_size in bytes / (bitrate in bytes per second)
            # 30 kbps = 3750 bytes/sec
            estimated_duration = file_size / 3750.0
            return estimated_duration
        except:
            return 0.0


def calculate_total_duration(audio_files: List[str]) -> float:
    """
    Calculate total duration of multiple audio files.

    Args:
        audio_files: List of audio file paths

    Returns:
        Total duration in seconds
    """
    total = 0.0
    for audio_file in audio_files:
        duration = get_audio_duration(audio_file)
        total += duration
    return total
