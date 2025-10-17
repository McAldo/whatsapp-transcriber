"""
Audio Processor
Handles transcription of audio files using faster-whisper.
"""

import os
import re
import logging
from typing import Optional, Callable, Dict, List
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class AudioTranscriber:
    """Handles audio transcription using faster-whisper."""

    def __init__(self, model_size: str = "base", device: str = "auto"):
        """
        Initialize the transcriber.

        Args:
            model_size: Size of whisper model (tiny, base, small, medium, large)
            device: Device to use (cpu, cuda, or auto)
        """
        self.model_size = model_size
        self.device = self._detect_device() if device == "auto" else device
        self.model = None
        self.gpu_error = None  # Store GPU error for UI display
        logger.info(f"AudioTranscriber initialized with model={model_size}, device={self.device}")

    def _detect_device(self) -> str:
        """Detect the best available device for transcription."""
        # Try to detect CUDA support via CTranslate2 (used by faster-whisper)
        try:
            import ctranslate2
            # Check if CUDA is available in CTranslate2
            cuda_available = ctranslate2.get_cuda_device_count() > 0
            if cuda_available:
                logger.info(f"CUDA available, using GPU ({ctranslate2.get_cuda_device_count()} device(s) detected)")
                return "cuda"
        except Exception as e:
            logger.debug(f"CUDA detection failed: {e}")

        logger.info("Using CPU for transcription (no CUDA devices found)")
        return "cpu"

    def load_model(self):
        """Load the faster-whisper model."""
        if self.model is not None:
            return  # Already loaded

        try:
            from faster_whisper import WhisperModel

            logger.info(f"Loading faster-whisper model: {self.model_size} on {self.device}")

            # Define compute type options in order of preference
            if self.device == "cuda":
                # Try multiple compute types for GPU compatibility
                compute_types = ["float16", "int8_float16", "int8"]
            else:
                compute_types = ["int8"]

            model_loaded = False
            last_error = None

            for compute_type in compute_types:
                try:
                    logger.info(f"Attempting to load with compute_type={compute_type}")

                    self.model = WhisperModel(
                        self.model_size,
                        device=self.device,
                        compute_type=compute_type,
                        download_root=None  # Use default cache location
                    )

                    logger.info(f"✓ Model loaded successfully on {self.device} (compute_type={compute_type})")
                    model_loaded = True
                    break  # Success! Exit loop

                except Exception as e:
                    last_error = e
                    error_msg = str(e).lower()

                    # Check if it's a compute type issue
                    if "compute type" in error_msg or "float16" in error_msg:
                        logger.warning(f"Compute type {compute_type} not supported, trying next option...")
                        continue
                    # Check for cuDNN/CUDA library issues
                    elif "cudnn" in error_msg or "cuda" in error_msg or ".dll" in error_msg:
                        logger.error(f"CUDA/cuDNN library issue detected: {e}")
                        logger.error("This likely means cuDNN is not properly installed")
                        # Don't try other compute types, this is a library issue
                        break
                    else:
                        # Different error, don't continue trying
                        raise

            # If all GPU compute types failed, fall back to CPU
            if not model_loaded and self.device == "cuda":
                error_str = str(last_error)

                # Determine error type for better user feedback
                if "cudnn" in error_str.lower() or "cuda" in error_str.lower():
                    self.gpu_error = "cudnn_missing"
                    logger.warning(f"GPU failed due to CUDA/cuDNN library issues: {last_error}")
                else:
                    self.gpu_error = "compute_type"
                    logger.warning(f"All GPU compute types failed. Last error: {last_error}")

                logger.info("Falling back to CPU...")

                self.device = "cpu"

                self.model = WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type="int8",
                    download_root=None
                )

                logger.info(f"✓ Model loaded successfully on CPU (fallback)")
                model_loaded = True

            if not model_loaded:
                raise RuntimeError(f"Failed to load model with any compute type. Last error: {last_error}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def transcribe_file(self, audio_path: str, language: Optional[str] = None) -> Dict[str, any]:
        """
        Transcribe a single audio file.

        Args:
            audio_path: Path to audio file
            language: Optional language code (e.g., 'en', 'es'). Auto-detect if None.

        Returns:
            Dictionary with transcription results
        """
        if self.model is None:
            self.load_model()

        try:
            logger.info(f"Transcribing: {audio_path}")

            # Transcribe
            segments, info = self.model.transcribe(
                audio_path,
                language=language,
                beam_size=5,
                vad_filter=True,  # Use voice activity detection
                vad_parameters=dict(
                    min_silence_duration_ms=500
                )
            )

            # Collect all segments
            text_segments = []
            full_text = []

            for segment in segments:
                text_segments.append({
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text.strip()
                })
                full_text.append(segment.text.strip())

            result = {
                'text': ' '.join(full_text),
                'segments': text_segments,
                'language': info.language,
                'language_probability': info.language_probability,
                'duration': info.duration,
                'success': True,
                'error': None
            }

            logger.info(f"Transcription complete: {len(text_segments)} segments, "
                       f"language={info.language} ({info.language_probability:.2f})")

            return result

        except Exception as e:
            logger.error(f"Transcription failed for {audio_path}: {e}")
            return {
                'text': '',
                'segments': [],
                'language': None,
                'language_probability': 0.0,
                'duration': 0.0,
                'success': False,
                'error': str(e)
            }

    def transcribe_batch(
        self,
        audio_files: List[str],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        language: Optional[str] = None
    ) -> Dict[str, Dict]:
        """
        Transcribe multiple audio files.

        Args:
            audio_files: List of audio file paths
            progress_callback: Optional callback function(current, total, filename)
            language: Optional language code

        Returns:
            Dictionary mapping filenames to transcription results
        """
        results = {}
        total = len(audio_files)

        for idx, audio_file in enumerate(audio_files, 1):
            filename = os.path.basename(audio_file)

            if progress_callback:
                progress_callback(idx, total, filename)

            result = self.transcribe_file(audio_file, language)
            results[audio_file] = result

        return results

    def extract_timestamp_from_filename(self, filename: str) -> Optional[datetime]:
        """
        Extract timestamp from WhatsApp audio filename.

        WhatsApp audio files are typically named: PTT-YYYYMMDD-WA####.opus

        Args:
            filename: Audio filename

        Returns:
            datetime object or None if can't parse
        """
        # Pattern: PTT-YYYYMMDD-WA####.opus
        match = re.search(r'PTT-(\d{8})-WA\d+', filename)
        if match:
            date_str = match.group(1)
            try:
                # Parse YYYYMMDD
                date = datetime.strptime(date_str, '%Y%m%d')
                logger.debug(f"Extracted date from {filename}: {date}")
                return date
            except ValueError as e:
                logger.warning(f"Could not parse date from {filename}: {e}")

        # Try to get file modification time as fallback
        return None

    def get_file_creation_time(self, filepath: str) -> Optional[datetime]:
        """
        Get file creation/modification time as fallback for timestamp.

        Args:
            filepath: Path to file

        Returns:
            datetime object or None
        """
        try:
            stat = os.stat(filepath)
            # Use creation time if available (Windows), otherwise modification time
            timestamp = stat.st_ctime if hasattr(stat, 'st_ctime') else stat.st_mtime
            return datetime.fromtimestamp(timestamp)
        except Exception as e:
            logger.warning(f"Could not get file time for {filepath}: {e}")
            return None

    def unload_model(self):
        """Unload model to free memory."""
        if self.model is not None:
            logger.info("Unloading transcription model")
            del self.model
            self.model = None

            # Try to free GPU memory if using CUDA
            if self.device == "cuda":
                try:
                    import torch
                    torch.cuda.empty_cache()
                except ImportError:
                    pass


def is_model_cached(model_size: str) -> bool:
    """
    Check if a Whisper model is already cached locally.

    Args:
        model_size: Model size name

    Returns:
        True if model is cached, False otherwise
    """
    try:
        from faster_whisper.utils import download_model
        import os

        # Get default cache directory
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")

        # Model name format used by faster-whisper
        model_names = [
            f"models--guillaumekln--faster-whisper-{model_size}",
            f"models--Systran--faster-whisper-{model_size}"
        ]

        # Check if any of the model directories exist
        for model_name in model_names:
            model_path = os.path.join(cache_dir, model_name)
            if os.path.exists(model_path):
                return True

        return False

    except Exception:
        # If we can't determine, assume not cached (safer)
        return False


def get_model_info(model_size: str) -> Dict[str, str]:
    """
    Get information about a Whisper model size.

    Args:
        model_size: Model size name

    Returns:
        Dictionary with model information
    """
    models = {
        'tiny': {
            'size': '~75 MB',
            'speed': 'Very Fast',
            'quality': 'Basic',
            'description': 'Fastest but lowest quality. Good for clear audio.'
        },
        'base': {
            'size': '~150 MB',
            'speed': 'Fast',
            'quality': 'Good',
            'description': 'Recommended for quick processing with decent quality.'
        },
        'small': {
            'size': '~500 MB',
            'speed': 'Medium',
            'quality': 'Very Good',
            'description': 'Best balance of speed and quality for most uses.'
        },
        'medium': {
            'size': '~1.5 GB',
            'speed': 'Slow',
            'quality': 'Excellent',
            'description': 'High quality but slower. Good for important transcriptions.'
        },
        'large': {
            'size': '~3 GB',
            'speed': 'Very Slow',
            'quality': 'Best',
            'description': 'Best quality but very slow. Use only when accuracy is critical.'
        }
    }

    return models.get(model_size, {
        'size': 'Unknown',
        'speed': 'Unknown',
        'quality': 'Unknown',
        'description': 'Unknown model size'
    })


def get_gpu_status() -> Dict[str, any]:
    """
    Get GPU availability status for transcription.

    Returns:
        Dictionary with GPU status information
    """
    try:
        import ctranslate2

        cuda_count = ctranslate2.get_cuda_device_count()

        if cuda_count > 0:
            # Get GPU device names if possible
            device_names = []
            try:
                import pynvml
                pynvml.nvmlInit()
                for i in range(cuda_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle)
                    device_names.append(name.decode('utf-8') if isinstance(name, bytes) else name)
                pynvml.nvmlShutdown()
            except:
                # If pynvml not available, just show count
                device_names = [f"CUDA Device {i}" for i in range(cuda_count)]

            return {
                'available': True,
                'device_count': cuda_count,
                'device_names': device_names,
                'backend': 'CTranslate2 (CUDA)',
                'message': f"✓ GPU acceleration available ({cuda_count} device{'s' if cuda_count > 1 else ''})",
                'note': 'Will auto-select best compute type (float16/int8_float16/int8)'
            }
        else:
            return {
                'available': False,
                'device_count': 0,
                'device_names': [],
                'backend': 'CTranslate2 (CPU)',
                'message': "⚠️ No GPU detected - using CPU (slower)"
            }

    except ImportError:
        return {
            'available': False,
            'device_count': 0,
            'device_names': [],
            'backend': 'Unknown',
            'message': "⚠️ CTranslate2 not found - check installation"
        }
    except Exception as e:
        return {
            'available': False,
            'device_count': 0,
            'device_names': [],
            'backend': 'Error',
            'message': f"⚠️ GPU detection error: {str(e)}"
        }
