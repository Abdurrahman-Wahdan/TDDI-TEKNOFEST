"""
Token usage extraction utilities for different LLM providers.

Simple utility to extract input tokens, output tokens, and total tokens
from individual LLM provider responses.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def extract_anthropic_tokens(response: Any) -> Dict[str, int]:
    """
    Extract token usage from Anthropic/Claude response.
    
    Returns:
        Dict with 'input_tokens', 'output_tokens', and 'total_tokens'
    """
    try:
        usage_data = None
        
        # Try different response formats
        if hasattr(response, 'response_metadata') and response.response_metadata:
            usage_data = response.response_metadata.get('usage', {})
        elif hasattr(response, 'usage') and response.usage:
            usage_data = response.usage
        elif hasattr(response, 'llm_output') and response.llm_output:
            usage_data = response.llm_output.get('usage', {})
        elif isinstance(response, dict) and 'usage' in response:
            usage_data = response['usage']
        
        if usage_data:
            # Handle both dict and object formats
            if hasattr(usage_data, 'input_tokens'):
                input_tokens = getattr(usage_data, 'input_tokens', 0)
                output_tokens = getattr(usage_data, 'output_tokens', 0)
            else:
                input_tokens = usage_data.get('input_tokens', 0)
                output_tokens = usage_data.get('output_tokens', 0)
            
            total_tokens = input_tokens + output_tokens
            
            logger.debug(f"Extracted Anthropic tokens: input={input_tokens}, output={output_tokens}, total={total_tokens}")
            
            return {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens
            }
        
    except Exception as e:
        logger.warning(f"Failed to extract Anthropic tokens: {e}")
    
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def extract_openai_tokens(response: Any) -> Dict[str, int]:
    """
    Extract token usage from OpenAI response.
    
    Returns:
        Dict with 'input_tokens', 'output_tokens', and 'total_tokens'
    """
    try:
        usage_data = None
        
        # Try different response formats
        if hasattr(response, 'response_metadata') and response.response_metadata:
            usage_data = response.response_metadata.get('token_usage', {})
        elif hasattr(response, 'usage') and response.usage:
            usage_data = response.usage
        elif hasattr(response, 'llm_output') and response.llm_output:
            usage_data = response.llm_output.get('token_usage', {})
        elif isinstance(response, dict) and 'usage' in response:
            usage_data = response['usage']
        
        if usage_data:
            # Handle both dict and object formats
            if hasattr(usage_data, 'prompt_tokens'):
                input_tokens = getattr(usage_data, 'prompt_tokens', 0)
                output_tokens = getattr(usage_data, 'completion_tokens', 0)
                total_tokens = getattr(usage_data, 'total_tokens', input_tokens + output_tokens)
            else:
                input_tokens = usage_data.get('prompt_tokens', 0)
                output_tokens = usage_data.get('completion_tokens', 0)
                total_tokens = usage_data.get('total_tokens', input_tokens + output_tokens)
            
            logger.debug(f"Extracted OpenAI tokens: input={input_tokens}, output={output_tokens}, total={total_tokens}")
            
            return {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens
            }
        
    except Exception as e:
        logger.warning(f"Failed to extract OpenAI tokens: {e}")
    
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def extract_gemini_tokens(response: Any) -> Dict[str, int]:
    """
    Extract token usage from Google Gemini response.
    
    Returns:
        Dict with 'input_tokens', 'output_tokens', and 'total_tokens'
    """
    try:
        usage_data = None
        
        # Try different response formats
        if hasattr(response, 'response_metadata') and response.response_metadata:
            usage_data = response.response_metadata.get('usage_metadata', {})
        elif hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage_data = response.usage_metadata
        elif hasattr(response, 'llm_output') and response.llm_output:
            usage_data = response.llm_output.get('usage_metadata', {})
        elif isinstance(response, dict) and 'usage_metadata' in response:
            usage_data = response['usage_metadata']
        
        if usage_data:
            # Handle both dict and object formats
            if hasattr(usage_data, 'prompt_token_count'):
                input_tokens = getattr(usage_data, 'prompt_token_count', 0)
                output_tokens = getattr(usage_data, 'candidates_token_count', 0)
                total_tokens = getattr(usage_data, 'total_token_count', input_tokens + output_tokens)
            else:
                input_tokens = usage_data.get('prompt_token_count', 0)
                output_tokens = usage_data.get('candidates_token_count', 0)
                total_tokens = usage_data.get('total_token_count', input_tokens + output_tokens)
            
            logger.debug(f"Extracted Gemini tokens: input={input_tokens}, output={output_tokens}, total={total_tokens}")
            
            return {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens
            }
        
    except Exception as e:
        logger.warning(f"Failed to extract Gemini tokens: {e}")
    
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def extract_azure_openai_tokens(response: Any) -> Dict[str, int]:
    """
    Extract token usage from Azure OpenAI response.
    Azure OpenAI uses the same format as regular OpenAI.
    
    Returns:
        Dict with 'input_tokens', 'output_tokens', and 'total_tokens'
    """
    return extract_openai_tokens(response)


def extract_local_tokens(response: Any) -> Dict[str, int]:
    """
    Extract token usage from local LLM response.
    Local LLMs may not provide token usage, so we return zero.
    
    Returns:
        Dict with 'input_tokens', 'output_tokens', and 'total_tokens'
    """
    try:
        usage_data = None
        
        if hasattr(response, 'response_metadata') and response.response_metadata:
            usage_data = response.response_metadata.get('usage', {})
        elif hasattr(response, 'usage') and response.usage:
            usage_data = response.usage
        elif isinstance(response, dict) and 'usage' in response:
            usage_data = response['usage']
        
        if usage_data:
            # Handle various formats that local LLMs might use
            input_tokens = usage_data.get('prompt_tokens', usage_data.get('input_tokens', 0))
            output_tokens = usage_data.get('completion_tokens', usage_data.get('output_tokens', 0))
            total_tokens = usage_data.get('total_tokens', input_tokens + output_tokens)
            
            logger.debug(f"Extracted Local LLM tokens: input={input_tokens}, output={output_tokens}, total={total_tokens}")
            
            return {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens
            }
        
    except Exception as e:
        logger.warning(f"Failed to extract Local LLM tokens: {e}")
    
    # Local LLMs often don't provide token counts, so return zero
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def extract_tokens_by_provider(provider: str, response: Any) -> Dict[str, int]:
    """
    Extract tokens based on provider type.
    
    Args:
        provider: Provider name ('anthropic', 'openai', 'gemini', 'azure_openai', 'local')
        response: LLM response object
        
    Returns:
        Dict with 'input_tokens', 'output_tokens', and 'total_tokens'
    """
    provider_lower = provider.lower()
    
    if provider_lower == 'anthropic':
        return extract_anthropic_tokens(response)
    elif provider_lower == 'openai':
        return extract_openai_tokens(response)
    elif provider_lower == 'gemini':
        return extract_gemini_tokens(response)
    elif provider_lower == 'azure_openai':
        return extract_azure_openai_tokens(response)
    elif provider_lower == 'local':
        return extract_local_tokens(response)
    else:
        logger.warning(f"Unknown provider: {provider}, returning zero tokens")
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


# Convenience function (same as extract_tokens_by_provider)
def extract_tokens(provider: str, response: Any) -> Dict[str, int]:
    """
    Main function to extract tokens from any provider response.
    
    Args:
        provider: Provider name
        response: LLM response object
        
    Returns:
        Dict with 'input_tokens', 'output_tokens', and 'total_tokens'
    """
    return extract_tokens_by_provider(provider, response)

