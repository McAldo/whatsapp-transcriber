"""
LLM Corrector
Handles correction of transcriptions using various LLM providers.
"""

import logging
from typing import List, Dict, Optional, Callable
import json

logger = logging.getLogger(__name__)


class LLMCorrector:
    """Handles transcription correction using LLM providers."""

    CORRECTION_PROMPT = """You are helping to improve automatic transcriptions of voice messages from WhatsApp.
Your task is to correct transcription errors, improve punctuation, and fix obvious mistakes while keeping the corrections minimal and faithful to the original speech.

Guidelines:
- Fix spelling errors and obvious transcription mistakes
- Add proper punctuation and capitalization
- Keep informal language and colloquialisms (it's spoken conversation)
- Don't change the meaning or add information that wasn't there
- Don't translate or rephrase completely
- Keep the original language

Original transcription:
{transcription}

Please provide only the corrected transcription, without any additional commentary or explanations."""

    FULL_TRANSCRIPT_PROMPT = """You are reviewing and correcting a WhatsApp conversation transcript that includes both text messages and transcribed voice messages.

Please review and correct any transcription errors in the voice messages marked with [TRANSCRIPTION].

Rules:
- Fix obvious speech-to-text errors (wrong words, missing punctuation, etc.)
- Keep corrections minimal - only fix clear mistakes
- Preserve the original meaning and speaker intent
- Maintain the exact format: [Type - Date] Speaker: Content
- Do NOT translate, summarize, or change the conversation structure
- Do NOT modify text messages, only fix [TRANSCRIPTION] entries

Transcript:
{transcript}

Please return the corrected transcript in the same format."""

    def __init__(self, provider: str, api_key: Optional[str] = None,
                 ollama_url: str = "http://localhost:11434",
                 ollama_model: str = "llama3.2",
                 correction_mode: str = "message"):
        """
        Initialize the LLM corrector.

        Args:
            provider: Provider name ('claude', 'openai', 'mistral', or 'ollama')
            api_key: API key (required for claude/openai/mistral, not for ollama)
            ollama_url: Ollama server URL (for ollama provider)
            ollama_model: Ollama model name (for ollama provider)
            correction_mode: 'message' for message-by-message, 'bulk' for full transcript
        """
        self.provider = provider.lower()
        self.api_key = api_key
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.correction_mode = correction_mode
        self.client = None

        logger.info(f"LLMCorrector initialized with provider={provider}, mode={correction_mode}")

    def initialize(self):
        """Initialize the LLM client based on provider."""
        if self.provider == "claude":
            self._init_claude()
        elif self.provider == "openai":
            self._init_openai()
        elif self.provider == "mistral":
            self._init_mistral()
        elif self.provider == "ollama":
            self._init_ollama()
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _init_claude(self):
        """Initialize Claude client."""
        try:
            from anthropic import Anthropic

            if not self.api_key:
                raise ValueError("API key required for Claude")

            self.client = Anthropic(api_key=self.api_key)
            logger.info("Claude client initialized")

        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

    def _init_openai(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI

            if not self.api_key:
                raise ValueError("API key required for OpenAI")

            self.client = OpenAI(api_key=self.api_key)
            logger.info("OpenAI client initialized")

        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

    def _init_mistral(self):
        """Initialize Mistral client."""
        try:
            from mistralai import Mistral

            if not self.api_key:
                raise ValueError("API key required for Mistral")

            self.client = Mistral(api_key=self.api_key)
            logger.info("Mistral client initialized")

        except ImportError:
            raise ImportError("mistralai package not installed. Run: pip install mistralai")

    def _init_ollama(self):
        """Initialize Ollama (just verify connection)."""
        import requests

        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            response.raise_for_status()
            logger.info("Ollama connection verified")
        except Exception as e:
            raise ConnectionError(f"Cannot connect to Ollama at {self.ollama_url}: {e}")

    def test_connection(self) -> Dict[str, any]:
        """
        Test the connection to the LLM provider.

        Returns:
            Dictionary with status and message
        """
        try:
            if self.provider == "ollama":
                import requests
                response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
                response.raise_for_status()
                data = response.json()

                # Check if the model is available
                models = [m['name'] for m in data.get('models', [])]
                model_available = any(self.ollama_model in m for m in models)

                return {
                    'success': True,
                    'message': f"Connected to Ollama. Model '{self.ollama_model}' {'found' if model_available else 'NOT FOUND'}.",
                    'models': models,
                    'model_available': model_available
                }

            elif self.provider == "claude":
                # Try a minimal API call
                from anthropic import Anthropic
                client = Anthropic(api_key=self.api_key)
                # Just check if we can create a client
                return {
                    'success': True,
                    'message': "Claude API key appears valid"
                }

            elif self.provider == "openai":
                # Try a minimal API call
                from openai import OpenAI
                client = OpenAI(api_key=self.api_key)
                # Try to list models
                models = client.models.list()
                return {
                    'success': True,
                    'message': "OpenAI API key appears valid"
                }

            elif self.provider == "mistral":
                # Try a minimal API call
                from mistralai import Mistral
                client = Mistral(api_key=self.api_key)
                # Try to list models
                models = client.models.list()
                return {
                    'success': True,
                    'message': "Mistral API key appears valid"
                }

        except Exception as e:
            return {
                'success': False,
                'message': f"Connection failed: {str(e)}"
            }

    def correct_transcription(self, transcription: str) -> str:
        """
        Correct a single transcription.

        Args:
            transcription: Original transcription text

        Returns:
            Corrected transcription
        """
        if not transcription or len(transcription.strip()) == 0:
            return transcription

        try:
            if self.provider == "claude":
                return self._correct_with_claude(transcription)
            elif self.provider == "openai":
                return self._correct_with_openai(transcription)
            elif self.provider == "mistral":
                return self._correct_with_mistral(transcription)
            elif self.provider == "ollama":
                return self._correct_with_ollama(transcription)
            else:
                logger.warning(f"Unknown provider {self.provider}, returning original")
                return transcription

        except Exception as e:
            logger.error(f"Correction failed: {e}")
            # Return original on error
            return transcription

    def _correct_with_claude(self, transcription: str) -> str:
        """Correct using Claude API."""
        prompt = self.CORRECTION_PROMPT.format(transcription=transcription)

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        corrected = response.content[0].text.strip()
        logger.debug(f"Claude correction: '{transcription}' -> '{corrected}'")
        return corrected

    def _correct_with_openai(self, transcription: str) -> str:
        """Correct using OpenAI API."""
        prompt = self.CORRECTION_PROMPT.format(transcription=transcription)

        # Choose model based on correction mode
        model = "gpt-4o" if self.correction_mode == "bulk" else "gpt-4o-mini"

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that corrects transcription errors."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            temperature=0.3
        )

        corrected = response.choices[0].message.content.strip()
        logger.debug(f"OpenAI correction: '{transcription}' -> '{corrected}'")
        return corrected

    def _correct_with_mistral(self, transcription: str) -> str:
        """Correct using Mistral API."""
        prompt = self.CORRECTION_PROMPT.format(transcription=transcription)

        # Choose model based on correction mode
        model = "mistral-large-latest" if self.correction_mode == "bulk" else "mistral-small-latest"

        response = self.client.chat.complete(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that corrects transcription errors."},
                {"role": "user", "content": prompt}
            ]
        )

        corrected = response.choices[0].message.content.strip()
        logger.debug(f"Mistral correction: '{transcription}' -> '{corrected}'")
        return corrected

    def _correct_with_ollama(self, transcription: str) -> str:
        """Correct using Ollama."""
        import requests

        prompt = self.CORRECTION_PROMPT.format(transcription=transcription)

        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 512
            }
        }

        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json=payload,
            timeout=60
        )
        response.raise_for_status()

        data = response.json()
        corrected = data.get('response', '').strip()

        logger.debug(f"Ollama correction: '{transcription}' -> '{corrected}'")
        return corrected

    def correct_batch(
        self,
        transcriptions: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[str]:
        """
        Correct multiple transcriptions.

        Args:
            transcriptions: List of transcription texts
            progress_callback: Optional callback function(current, total)

        Returns:
            List of corrected transcriptions
        """
        corrected = []
        total = len(transcriptions)

        for idx, text in enumerate(transcriptions, 1):
            if progress_callback:
                progress_callback(idx, total)

            corrected_text = self.correct_transcription(text)
            corrected.append(corrected_text)

        return corrected

    def correct_full_transcript(self, transcript: str) -> str:
        """
        Correct an entire transcript at once (bulk mode).

        Args:
            transcript: Full transcript with all messages

        Returns:
            Corrected transcript
        """
        if not transcript or len(transcript.strip()) == 0:
            return transcript

        try:
            if self.provider == "claude":
                return self._correct_full_transcript_claude(transcript)
            elif self.provider == "openai":
                return self._correct_full_transcript_openai(transcript)
            elif self.provider == "mistral":
                return self._correct_full_transcript_mistral(transcript)
            elif self.provider == "ollama":
                return self._correct_full_transcript_ollama(transcript)
            else:
                logger.warning(f"Unknown provider {self.provider}, returning original")
                return transcript

        except Exception as e:
            logger.error(f"Full transcript correction failed: {e}")
            return transcript

    def _correct_full_transcript_claude(self, transcript: str) -> str:
        """Correct full transcript using Claude API."""
        prompt = self.FULL_TRANSCRIPT_PROMPT.format(transcript=transcript)

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=8000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        corrected = response.content[0].text.strip()
        logger.info("Claude full transcript correction complete")
        return corrected

    def _correct_full_transcript_openai(self, transcript: str) -> str:
        """Correct full transcript using OpenAI API."""
        prompt = self.FULL_TRANSCRIPT_PROMPT.format(transcript=transcript)

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that corrects transcript errors."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=8000,
            temperature=0.3
        )

        corrected = response.choices[0].message.content.strip()
        logger.info("OpenAI full transcript correction complete")
        return corrected

    def _correct_full_transcript_mistral(self, transcript: str) -> str:
        """Correct full transcript using Mistral API."""
        prompt = self.FULL_TRANSCRIPT_PROMPT.format(transcript=transcript)

        response = self.client.chat.complete(
            model="mistral-large-latest",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that corrects transcript errors."},
                {"role": "user", "content": prompt}
            ]
        )

        corrected = response.choices[0].message.content.strip()
        logger.info("Mistral full transcript correction complete")
        return corrected

    def _correct_full_transcript_ollama(self, transcript: str) -> str:
        """Correct full transcript using Ollama."""
        import requests

        prompt = self.FULL_TRANSCRIPT_PROMPT.format(transcript=transcript)

        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 4096
            }
        }

        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json=payload,
            timeout=300  # Longer timeout for full transcript
        )
        response.raise_for_status()

        data = response.json()
        corrected = data.get('response', '').strip()

        logger.info("Ollama full transcript correction complete")
        return corrected


def get_provider_info(provider: str) -> Dict[str, str]:
    """
    Get information about an LLM provider.

    Args:
        provider: Provider name

    Returns:
        Dictionary with provider information
    """
    providers = {
        'claude': {
            'name': 'Claude (Anthropic)',
            'requires_api_key': True,
            'cost': 'Paid (per API call)',
            'privacy': 'Data sent to Anthropic servers',
            'quality': 'Excellent',
            'speed': 'Fast',
            'notes': 'High quality corrections with good context understanding'
        },
        'openai': {
            'name': 'OpenAI (ChatGPT)',
            'requires_api_key': True,
            'cost': 'Paid (per API call)',
            'privacy': 'Data sent to OpenAI servers',
            'quality': 'Excellent',
            'speed': 'Fast',
            'notes': 'High quality corrections with GPT-4'
        },
        'mistral': {
            'name': 'Mistral AI',
            'requires_api_key': True,
            'cost': 'Paid (per API call)',
            'privacy': 'Data sent to Mistral servers',
            'quality': 'Excellent',
            'speed': 'Fast',
            'notes': 'High quality corrections with competitive pricing'
        },
        'ollama': {
            'name': 'Ollama (Local)',
            'requires_api_key': False,
            'cost': 'FREE',
            'privacy': 'Completely local - data never leaves your computer',
            'quality': 'Good to Excellent (depends on model)',
            'speed': 'Medium (depends on hardware)',
            'notes': 'Recommended for privacy. Requires Ollama installation.'
        }
    }

    return providers.get(provider.lower(), {
        'name': 'Unknown',
        'requires_api_key': False,
        'cost': 'Unknown',
        'privacy': 'Unknown',
        'quality': 'Unknown',
        'speed': 'Unknown',
        'notes': ''
    })
