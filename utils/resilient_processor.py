"""
Resilient Processor
Handles timeout wrappers and file info collection for robust processing.
"""

import os
import threading
import logging
from typing import Callable, Any, Tuple, Dict, Optional
from datetime import datetime
import time

logger = logging.getLogger(__name__)


def process_with_timeout(
    func: Callable,
    args: Tuple,
    timeout_seconds: int = 180,
    kwargs: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Execute function with timeout using threading.

    Args:
        func: Function to execute
        args: Tuple of positional arguments
        timeout_seconds: Maximum time to wait (default 180s = 3 minutes)
        kwargs: Optional keyword arguments

    Returns:
        Result dictionary with success/error status
    """
    if kwargs is None:
        kwargs = {}

    result = {"success": False, "error": "timeout"}
    start_time = time.time()

    def target():
        nonlocal result
        try:
            logger.debug(f"Starting function execution in thread")
            func_result = func(*args, **kwargs)

            # Handle different return types
            if isinstance(func_result, dict):
                result = func_result
            else:
                result = {"success": True, "result": func_result}

            elapsed = time.time() - start_time
            logger.debug(f"Function completed in {elapsed:.2f}s")

        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = str(e)
            logger.error(f"Function raised exception after {elapsed:.2f}s: {error_msg}")
            result = {
                "success": False,
                "error": error_msg,
                "error_type": type(e).__name__
            }

    # Create and start thread
    thread = threading.Thread(target=target, daemon=True)
    thread.start()

    logger.info(f"Waiting up to {timeout_seconds}s for completion...")
    thread.join(timeout=timeout_seconds)

    elapsed = time.time() - start_time

    if thread.is_alive():
        # Timeout occurred
        logger.warning(f"⏱️ Timeout after {timeout_seconds}s (waited {elapsed:.2f}s)")
        return {
            "success": False,
            "error": f"timeout after {timeout_seconds}s",
            "error_type": "TimeoutError",
            "elapsed_seconds": elapsed
        }

    # Add elapsed time to result
    if 'elapsed_seconds' not in result:
        result['elapsed_seconds'] = elapsed

    return result


def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Collect information about an audio file.

    Args:
        file_path: Path to audio file

    Returns:
        Dictionary with file information:
        - size_bytes: File size in bytes
        - duration_seconds: Audio duration (if available)
        - exists: Whether file exists
        - extension: File extension
    """
    info = {
        "size_bytes": None,
        "duration_seconds": None,
        "exists": False,
        "extension": None,
        "file_path": file_path
    }

    try:
        # Check if file exists
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return info

        info['exists'] = True

        # Get file size
        info['size_bytes'] = os.path.getsize(file_path)

        # Get file extension
        info['extension'] = os.path.splitext(file_path)[1].lower()

        # Try to get audio duration
        info['duration_seconds'] = get_audio_duration(file_path)

        logger.debug(f"File info for {os.path.basename(file_path)}: "
                    f"{info['size_bytes']} bytes, "
                    f"{info['duration_seconds']}s duration")

        return info

    except Exception as e:
        logger.error(f"Error collecting file info for {file_path}: {e}")
        return info


def get_audio_duration(audio_path: str) -> Optional[float]:
    """
    Safely get audio duration in seconds.

    Tries multiple methods in order:
    1. soundfile (lightweight, fast)
    2. librosa (more robust)
    3. Returns None if all fail

    Args:
        audio_path: Path to audio file

    Returns:
        Duration in seconds, or None if unable to determine
    """
    # Try soundfile first (lightweight)
    try:
        import soundfile as sf
        info = sf.info(audio_path)
        duration = info.duration
        logger.debug(f"Got duration via soundfile: {duration:.2f}s")
        return duration
    except ImportError:
        logger.debug("soundfile not available, trying librosa")
    except Exception as e:
        logger.debug(f"soundfile failed: {e}, trying librosa")

    # Try librosa as fallback
    try:
        import librosa
        duration = librosa.get_duration(path=audio_path)
        logger.debug(f"Got duration via librosa: {duration:.2f}s")
        return duration
    except ImportError:
        logger.debug("librosa not available")
    except Exception as e:
        logger.debug(f"librosa failed: {e}")

    # If both fail, return None
    logger.debug(f"Could not determine audio duration for {audio_path}")
    return None


def calculate_total_duration(audio_files: list) -> float:
    """
    Calculate total duration of multiple audio files.

    Args:
        audio_files: List of audio file paths

    Returns:
        Total duration in seconds
    """
    total_duration = 0.0
    successful_count = 0

    for audio_file in audio_files:
        duration = get_audio_duration(audio_file)
        if duration is not None:
            total_duration += duration
            successful_count += 1

    if successful_count < len(audio_files):
        logger.warning(f"Could only determine duration for {successful_count}/{len(audio_files)} files")

    return total_duration


class ProcessingTimer:
    """Helper class for timing operations."""

    def __init__(self, operation_name: str):
        """
        Initialize timer.

        Args:
            operation_name: Name of operation being timed
        """
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        """Start timer."""
        self.start_time = time.time()
        logger.debug(f"⏱️ Starting: {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timer and log duration."""
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time

        if exc_type is None:
            logger.info(f"✓ Completed: {self.operation_name} in {elapsed:.2f}s")
        else:
            logger.error(f"✗ Failed: {self.operation_name} after {elapsed:.2f}s")

    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0

        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time


def retry_with_backoff(
    func: Callable,
    args: Tuple,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    kwargs: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Retry function with exponential backoff.

    Args:
        func: Function to execute
        args: Tuple of positional arguments
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        backoff_factor: Multiplier for delay between retries
        kwargs: Optional keyword arguments

    Returns:
        Result dictionary with success/error status
    """
    if kwargs is None:
        kwargs = {}

    delay = initial_delay

    for attempt in range(1, max_retries + 1):
        logger.info(f"Attempt {attempt}/{max_retries}")

        try:
            result = func(*args, **kwargs)

            # Check if result indicates success
            if isinstance(result, dict):
                if result.get('success', True):
                    logger.info(f"✓ Succeeded on attempt {attempt}")
                    return result
                else:
                    error = result.get('error', 'Unknown error')
                    logger.warning(f"Attempt {attempt} failed: {error}")

                    if attempt < max_retries:
                        logger.info(f"Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        delay *= backoff_factor
                        continue
                    else:
                        result['attempts'] = attempt
                        return result
            else:
                # Non-dict result, assume success
                return {"success": True, "result": result, "attempts": attempt}

        except Exception as e:
            logger.error(f"Attempt {attempt} raised exception: {e}")

            if attempt < max_retries:
                logger.info(f"Retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay *= backoff_factor
            else:
                return {
                    "success": False,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "attempts": attempt
                }

    # Shouldn't reach here, but just in case
    return {
        "success": False,
        "error": "Max retries exceeded",
        "attempts": max_retries
    }


def log_processing_metrics(
    filename: str,
    file_info: Dict[str, Any],
    result: Dict[str, Any],
    elapsed_time: float
):
    """
    Log comprehensive metrics for a processed file.

    Args:
        filename: Name of file processed
        file_info: File information dictionary
        result: Processing result dictionary
        elapsed_time: Time taken to process
    """
    success = result.get('success', False)
    status_symbol = "✓" if success else "✗"

    log_msg = (
        f"{status_symbol} {filename}: "
        f"{elapsed_time:.2f}s, "
        f"{file_info.get('size_bytes', 0) / 1024:.1f}KB"
    )

    if file_info.get('duration_seconds'):
        log_msg += f", {file_info['duration_seconds']:.1f}s audio"

    if success:
        text_len = len(result.get('text', ''))
        log_msg += f", {text_len} chars transcribed"

        cost = result.get('cost', 0)
        if cost > 0:
            log_msg += f", ${cost:.4f}"
    else:
        error = result.get('error', 'Unknown error')
        log_msg += f" - ERROR: {error}"

    logger.info(log_msg)
