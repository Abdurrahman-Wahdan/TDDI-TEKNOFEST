"""
Simplified GEMMA-3-27B Model Provider for LangGraph Agents

This module provides a streamlined interface for using GEMMA-3-27B model
across multiple agents in a LangGraph-based system.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class GemmaConfig(BaseModel):
    """Configuration for GEMMA model"""
    model_name: str = Field(default="models/gemma-3-27b-it", description="GEMMA model name")
    api_key: Optional[str] = Field(default=None, description="Google AI API key")
    temperature: float = Field(default=0.1, ge=0.0, le=1.0, description="Model temperature")
    max_tokens: Optional[int] = Field(default=8192, description="Maximum tokens to generate")
    timeout: Optional[float] = Field(default=60.0, description="Request timeout in seconds")


class GemmaProvider:
    """
    Simplified GEMMA-3-27B provider for LangGraph agents.
    
    This class handles all GEMMA model interactions and provides a clean
    interface for agents to use the model with different configurations.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern to ensure one provider instance"""
        if cls._instance is None:
            cls._instance = super(GemmaProvider, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the GEMMA provider"""
        if self._initialized:
            return
            
        self.config = GemmaConfig()
        self.api_key = self._get_api_key()
        self._base_model = None
        self._initialized = True
        
        logger.info("GEMMA Provider initialized successfully")
    
    def _get_api_key(self) -> str:
        """Get API key from environment variables"""
        # Try different possible environment variable names
        api_key = (
            os.getenv("GEMMA_API_KEY") or 
            os.getenv("GOOGLE_API_KEY") or 
            os.getenv("GEMINI_API_KEY") or
            self.config.api_key
        )
        
        if not api_key:
            raise ValueError(
                "GEMMA API key not found. Please set one of: "
                "GEMMA_API_KEY, GOOGLE_API_KEY, or GEMINI_API_KEY"
            )
        
        return api_key
    
    def create_model(
        self, 
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model_name: Optional[str] = None,
        **kwargs
    ) -> ChatGoogleGenerativeAI:
        """
        Create a GEMMA model instance with specified configuration.
        
        Args:
            temperature: Model temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            model_name: Model name (defaults to gemma-3-27b-it)
            **kwargs: Additional model parameters
            
        Returns:
            ChatGoogleGenerativeAI: Configured GEMMA model instance
        """
        model_config = {
            "model": model_name or self.config.model_name,
            "google_api_key": self.api_key,
            "temperature": temperature or self.config.temperature,
            "max_output_tokens": max_tokens or self.config.max_tokens,
            "timeout": self.config.timeout,
            **kwargs
        }
        
        return ChatGoogleGenerativeAI(**model_config)
    
    def get_default_model(self) -> ChatGoogleGenerativeAI:
        """Get a default configured GEMMA model instance"""
        if self._base_model is None:
            self._base_model = self.create_model()
        return self._base_model
    
    def create_chat_prompt(
        self, 
        system_message: str, 
        human_template: str = "{input}",
        additional_variables: Optional[List[str]] = None
    ) -> ChatPromptTemplate:
        """
        Create a chat prompt template for agent interactions.
        
        Args:
            system_message: System message defining agent behavior
            human_template: Template for human messages
            additional_variables: Additional template variables
            
        Returns:
            ChatPromptTemplate: Configured prompt template
        """
        messages = [
            SystemMessagePromptTemplate.from_template(system_message),
            HumanMessagePromptTemplate.from_template(human_template)
        ]
        
        input_variables = ["input"]
        if additional_variables:
            input_variables.extend(additional_variables)
        
        return ChatPromptTemplate.from_messages(messages)
    
    def invoke_model(
        self, 
        prompt: str, 
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Simple method to invoke the model with a prompt.
        
        Args:
            prompt: The prompt to send to the model
            system_message: Optional system message
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            
        Returns:
            str: Model response
        """
        model = self.create_model(temperature=temperature, max_tokens=max_tokens)
        
        full_prompt = []
        if system_message:
            full_prompt.append(system_message)
        full_prompt.append(prompt)
        
        # Combine into a single prompt for the HumanMessage
        messages = [HumanMessage(content="\n\n".join(full_prompt))]
        
        response = model.invoke(messages)
        return response.content
    
    async def ainvoke_model(
        self, 
        prompt: str, 
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Async method to invoke the model with a prompt.
        
        Args:
            prompt: The prompt to send to the model
            system_message: Optional system message
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            
        Returns:
            str: Model response
        """
        model = self.create_model(temperature=temperature, max_tokens=max_tokens)
        
        full_prompt = []
        if system_message:
            full_prompt.append(system_message)
        full_prompt.append(prompt)
        
        # Combine into a single prompt for the HumanMessage
        messages = [HumanMessage(content="\n\n".join(full_prompt))]
        
        response = await model.ainvoke(messages)
        return response.content


# Global provider instance
gemma_provider = GemmaProvider()


def get_gemma_model(**kwargs) -> ChatGoogleGenerativeAI:
    """
    Convenience function to get a GEMMA model instance.
    
    Args:
        **kwargs: Model configuration parameters
        
    Returns:
        ChatGoogleGenerativeAI: Configured model instance
    """
    return gemma_provider.create_model(**kwargs)


def create_prompt_template(system_message: str, **kwargs) -> ChatPromptTemplate:
    """
    Convenience function to create a prompt template.
    
    Args:
        system_message: System message for the template
        **kwargs: Additional template parameters
        
    Returns:
        ChatPromptTemplate: Configured prompt template
    """
    return gemma_provider.create_chat_prompt(system_message, **kwargs)


def quick_invoke(prompt: str, system_message: Optional[str] = None, **kwargs) -> str:
    """
    Quick function to invoke GEMMA with a simple prompt.
    
    Args:
        prompt: The prompt to send
        system_message: Optional system message
        **kwargs: Model parameters
        
    Returns:
        str: Model response
    """
    return gemma_provider.invoke_model(prompt, system_message, **kwargs)


async def quick_ainvoke(prompt: str, system_message: Optional[str] = None, **kwargs) -> str:
    """
    Quick async function to invoke GEMMA with a simple prompt.
    
    Args:
        prompt: The prompt to send
        system_message: Optional system message
        **kwargs: Model parameters
        
    Returns:
        str: Model response
    """
    return await gemma_provider.ainvoke_model(prompt, system_message, **kwargs)


if __name__ == "__main__":
    # Example usage
    try:
        model = get_gemma_model()
        response = quick_invoke("What is the weather like today?", system_message="You are a helpful assistant.")
        print(f"Model Response: {response}")
    except Exception as e:
        logger.error(f"Error invoking GEMMA model: {e}")