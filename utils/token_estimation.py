"""
Token Estimation Utilities
Provides token counting for different LLM providers.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def estimate_tokens(text: str, provider: str = "claude") -> int:
    """
    Estimate token count for a given text and provider.

    Args:
        text: Text to estimate tokens for
        provider: LLM provider ('claude', 'openai', 'mistral', 'ollama')

    Returns:
        Estimated token count
    """
    if not text:
        return 0

    provider = provider.lower()

    try:
        if provider == "openai":
            # Use tiktoken for accurate OpenAI token counting
            import tiktoken
            encoder = tiktoken.encoding_for_model("gpt-4")
            return len(encoder.encode(text))
        elif provider == "claude":
            # Claude/Anthropic: ~3.5 characters per token
            return int(len(text) / 3.5)
        elif provider == "mistral":
            # Mistral: ~4 characters per token
            return int(len(text) / 4)
        else:
            # Default estimate for unknown providers (including Ollama)
            return int(len(text) / 4)
    except Exception as e:
        logger.warning(f"Token estimation failed for {provider}: {e}. Using default estimate.")
        # Fallback to character-based estimate
        return int(len(text) / 4)


def format_token_count(token_count: int) -> str:
    """
    Format token count with thousands separator.

    Args:
        token_count: Number of tokens

    Returns:
        Formatted string (e.g., "15,234")
    """
    return f"{token_count:,}"


def estimate_cost(token_count: int, provider: str, model: str, mode: str = "message") -> float:
    """
    Estimate cost for token usage.

    Args:
        token_count: Number of tokens
        provider: LLM provider
        model: Specific model being used
        mode: 'message' for message-by-message or 'bulk' for full transcript

    Returns:
        Estimated cost in USD
    """
    provider = provider.lower()

    # Pricing per 1K tokens (approximate, as of 2025)
    pricing = {
        'claude': {
            'claude-3-5-sonnet': {'input': 0.003, 'output': 0.015},
            'claude-3-5-sonnet-20241022': {'input': 0.003, 'output': 0.015},
        },
        'openai': {
            'gpt-4o': {'input': 0.0025, 'output': 0.01},
            'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006},
            'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
        },
        'mistral': {
            'mistral-small-latest': {'input': 0.002, 'output': 0.006},
            'mistral-large-latest': {'input': 0.008, 'output': 0.024},
        },
        'ollama': {
            'default': {'input': 0, 'output': 0},  # Local, free
        }
    }

    try:
        if provider == 'ollama':
            return 0.0  # Ollama is local and free

        # Get pricing for provider/model
        provider_pricing = pricing.get(provider, {})
        model_pricing = provider_pricing.get(model, {})

        if not model_pricing:
            # Try to find a default model for the provider
            if provider_pricing:
                model_pricing = list(provider_pricing.values())[0]
            else:
                return 0.0

        # Estimate input/output token split
        # For corrections: input is larger, output is roughly the same size
        input_tokens = int(token_count * 0.9)  # 90% input
        output_tokens = int(token_count * 0.1)  # 10% output (corrections are minor)

        # Calculate cost
        input_cost = (input_tokens / 1000) * model_pricing.get('input', 0)
        output_cost = (output_tokens / 1000) * model_pricing.get('output', 0)

        return input_cost + output_cost

    except Exception as e:
        logger.warning(f"Cost estimation failed: {e}")
        return 0.0


def check_token_limit(token_count: int, provider: str, model: str) -> dict:
    """
    Check if token count exceeds model limits.

    Args:
        token_count: Number of tokens
        provider: LLM provider
        model: Specific model

    Returns:
        Dictionary with 'ok', 'warning', 'error' flags and messages
    """
    # Context window limits
    limits = {
        'claude': {
            'claude-3-5-sonnet': 200000,
            'claude-3-5-sonnet-20241022': 200000,
        },
        'openai': {
            'gpt-4o': 128000,
            'gpt-4o-mini': 128000,
            'gpt-4-turbo': 128000,
        },
        'mistral': {
            'mistral-small-latest': 32000,
            'mistral-large-latest': 128000,
        },
        'ollama': {
            'default': 32000,  # Varies by model
        }
    }

    provider = provider.lower()
    limit = limits.get(provider, {}).get(model, 100000)  # Default 100k if unknown

    if token_count > limit:
        return {
            'ok': False,
            'warning': False,
            'error': True,
            'message': f"❌ Error: Transcript too large ({format_token_count(token_count)} tokens). "
                      f"Model limit is {format_token_count(limit)} tokens. Use Message-by-Message mode.",
            'limit': limit
        }
    elif token_count > limit * 0.8:  # >80% of limit
        return {
            'ok': True,
            'warning': True,
            'error': False,
            'message': f"⚠️ Warning: Large transcript ({format_token_count(token_count)} tokens). "
                      f"Approaching model limit of {format_token_count(limit)} tokens.",
            'limit': limit
        }
    else:
        return {
            'ok': True,
            'warning': False,
            'error': False,
            'message': f"✓ Token count within limits ({format_token_count(token_count)} tokens)",
            'limit': limit
        }
