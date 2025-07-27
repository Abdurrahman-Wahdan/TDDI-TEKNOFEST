"""
AI-Powered JOLT Transformation Specification Generator

This module provides an intelligent agent that generates JOLT (JSON-to-JSON Transformation 
Language) specifications using Large Language Models. The agent analyzes source and target 
JSON structures to automatically create transformation rules, eliminating the need for 
manual JOLT specification writing.

Core Functionality:
- Automated JOLT specification generation from JSON examples
- LangChain integration for advanced AI orchestration
- Support for optional tools to enhance specification quality
- Structured output validation with Pydantic models
- Flexible model selection and configuration overrides

Architecture:
- TransformerAgent: Main AI agent using LangChain framework
- TransformationInput/Result: Data models for type safety
- LLM Integration: Supports multiple providers via model registry
- Tool Integration: Optional enhancement with validation/execution tools

Integration Points:
- JoltOrchestration: Primary consumer for workflow automation
- Model Registry: Dynamic model selection and configuration
- Tool System: Optional schema validation and execution tools
"""

import logging
import json
from typing import Dict, List, Optional, Any, Union, Callable, TypedDict
from pydantic import BaseModel, Field

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models.base import BaseChatModel
from langchain_core.tools import BaseTool
from langchain.agents import AgentExecutor
from langchain.agents.structured_chat.base import StructuredChatAgent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from llm_providers.interfaces.base_llm import LLMConfig, LLMProviderType
from core.interfaces.agent_base import BaseAgent, AgentResult
from config.app_config import get_config
from llm_providers.model_registry import (
    ModelPurpose,
    get_langchain_model,
)

logger = logging.getLogger(__name__)


class TransformationInput(BaseModel):
    """
    Pydantic model for validating JOLT transformation input parameters.
    
    Ensures type safety and validation for transformation requests by defining
    the expected structure and constraints for source JSON, target JSON, and
    additional context information used by the AI agent.
    
    Attributes:
        source_json: The input JSON data that needs transformation (required)
        target_json: Example of desired output format (optional if context provided)
        context: Additional instructions or requirements for transformation
    """
    source_json: Union[Dict[str, Any], List[Any]] = Field(
        ..., description="Source JSON that needs to be transformed"
    )
    target_json: Optional[Union[Dict[str, Any], List[Any]]] = Field(
        ..., description="Target JSON showing the desired output format"
    )
    context: Optional[str] = Field(
        None, description="Additional context for the transformation"
    )


class TransformationResult(TypedDict):
    """
    TypedDict defining the structure of JOLT transformation results.
    
    Standardizes the output format from the AI agent to ensure consistent
    structure across different LLM providers and model versions. Includes
    the generated JOLT specification, explanations, and confidence metrics.
    
    Attributes:
        jolt_spec: List of JOLT transformation operations (required)
        explanation: Human-readable description of the transformation logic
        target_json: Generated or refined target JSON structure (if created by agent)
        confidence: Agent's confidence level in the generated specification (0.0-1.0)
    """
    jolt_spec: List[Dict[str, Any]]
    explanation: str 
    target_json: Optional[Dict[str, Any]] 
    confidence: float 


class TransformerAgent(BaseAgent):
    """
    LLM-powered agent for automated JOLT transformation specification generation.
    
    This agent leverages Large Language Models to analyze JSON structures and
    automatically generate JOLT transformation specifications. It uses LangChain
    for LLM orchestration and supports optional tools for enhanced accuracy.
    The agent is designed to handle complex transformations including field
    renaming, restructuring, array operations, and conditional logic.
    
    Key Capabilities:
    - Automated analysis of source/target JSON structures
    - Generation of complete JOLT transformation specifications
    - Support for complex transformations (arrays, nested objects, conditionals)
    - Integration with validation and execution tools
    - Flexible model selection and configuration
    - Structured output with explanation and confidence metrics
    
    Architecture:
    - Inherits from BaseAgent for consistent interface
    - Uses LangChain AgentExecutor for LLM orchestration
    - Supports both direct LLM invocation and tool-enhanced workflows
    - Configurable system prompts and model parameters
    
    Attributes:
        tools: Optional LangChain tools for enhanced specification generation
        model_id: Specific model identifier or None for purpose-based selection
        config_overrides: Model configuration parameters (temperature, max_tokens, etc.)
        llm: Initialized LangChain model instance
        system_prompt: Prompt defining agent behavior and expertise
        agent_executor: LangChain executor managing LLM interactions
        transformation_config: Application configuration for transformations
    """
    
    def __init__(
        self,
        tools: Optional[List[BaseTool]] = None,
        model_id: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the TransformerAgent with specified configuration.
        
        Sets up the AI agent with the chosen LLM model, optional tools, and
        configuration parameters. Initializes LangChain components and prepares
        the agent for JOLT specification generation requests.
        
        Args:
            tools (Optional[List[BaseTool]]): LangChain tools for enhanced generation
                                            (e.g., schema validators, JOLT executors)
            model_id (Optional[str]): Specific model to use, or None for purpose-based
                                    selection from model registry
            config_overrides (Optional[Dict[str, Any]]): Model configuration overrides
                                                       (temperature, max_tokens, etc.)
            system_prompt (Optional[str]): Custom system prompt defining agent behavior,
                                         or None to use default JOLT expertise prompt
                                         
        Side Effects:
            - Initializes LLM model through model registry
            - Creates LangChain AgentExecutor for request handling
            - Loads transformation configuration from app config
            - Logs initialization details for debugging
        """
        self.tools = tools or []
        self.model_id = model_id
        self.config_overrides = config_overrides or {}
        
        # Load application configuration for transformation settings
        app_config = get_config()
        self.transformation_config = app_config.transformation
        
        # Initialize the underlying LLM model
        self.llm = self._initialize_llm()
        
        # Set up system prompt for JOLT expertise
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        
        # Create LangChain agent executor for request orchestration
        self.agent_executor = self._create_agent_executor()
        
        logger.info(f"Transformer Agent initialized with model_id: {self.model_id}")
    
    def _initialize_llm(self) -> BaseChatModel:
        """
        Initialize the underlying LLM model for the transformer agent.
        
        Uses the model registry to obtain a properly configured LangChain model
        instance. Supports both specific model selection and purpose-based
        selection for transformer tasks. Applies configuration overrides for
        fine-tuning model behavior.
        
        Returns:
            BaseChatModel: Configured LangChain model ready for JOLT generation
            
        Raises:
            Exception: If model initialization fails or model_id is invalid
        """
        if self.model_id:
            # Use specific model ID with configuration overrides
            return get_langchain_model(
                model_id=self.model_id, 
                config_overrides=self.config_overrides
            )
        else:
            # Use purpose-based model selection for transformer tasks
            return get_langchain_model(
                purpose=ModelPurpose.TRANSFORMER,
                config_overrides=self.config_overrides
            )
    
    def _get_default_system_prompt(self) -> str:
        """
        Generate the default system prompt defining JOLT expertise and behavior.
        
        Creates a comprehensive prompt that establishes the agent as a JOLT
        specification expert and provides detailed instructions for analyzing
        JSON structures and generating transformation specifications. The prompt
        emphasizes creating reusable specifications that work across different
        data instances with the same structure.
        
        Returns:
            str: Detailed system prompt defining agent capabilities and approach
        """
        return """
        You are a JOLT specification expert. JOLT (JSON to JSON Transformation Language) is a tool for transforming JSON documents into other JSON formats.
        
        Your task is to create a JOLT specification that will transform the provided source JSON into the target JSON format.
        If user ask about another content to you. You will generate kind response about 'why you cannot response to user question'
        Follow these steps:
        1. Identify the user shared the target JSON with you if not, use context and create target JSON according to the context. If target JSON is available DO NOT create target json.
        2. If the target JSON created by you shra it on your response with target_json: 
        3. Analyze the source and target JSON structures carefully
        4. Identify the transformations needed (shifts, default values, modifies, etc.)
        5. Create a complete, valid JOLT specification
        6. Verify the specification would produce the expected output
        7. Provide a clear explanation of how your specification works
        
        Your specification should handle:
        - Field renaming and restructuring
        - Array transformations
        - Value modifications
        - Conditional logic if needed


        The structure of the input json file will never change, but the data inside it may change. So analyze the structure thoroughly and create the JOLT Transformation specification according to this structure. 
        And Return your answer as a valid JSON array containing the JOLT specification.
        """
    
    def _create_agent_executor(self) -> AgentExecutor:
        """
        Create and configure a LangChain AgentExecutor for request processing.
        
        Builds the orchestration layer that manages interactions between the LLM,
        optional tools, and structured output generation. Uses StructuredChatAgent
        for handling complex prompts and tool interactions. Configures error handling
        and verbose logging for debugging transformation generation.
        
        Returns:
            AgentExecutor: Configured executor ready to process transformation requests
            
        Side Effects:
            - Binds tools to LLM if tools are provided
            - Configures prompt template with system message and placeholders
            - Enables verbose logging and parsing error handling
        """
        # Create prompt template with system instructions and input placeholders
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        if self.tools:
            # Configure LLM with tool binding for enhanced capabilities
            llm_with_tools = self.llm.bind_tools(self.tools)
            
            agent = StructuredChatAgent.from_llm_and_tools(
                llm=llm_with_tools,
                tools=self.tools,
                prompt=prompt
            )
            
            return AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=True
            )
        else:
            # Configure for direct LLM interaction without tools
            return AgentExecutor(
                agent=StructuredChatAgent.from_llm_and_tools(
                    llm=self.llm,
                    tools=[],
                    prompt=prompt
                ),
                tools=[],
                verbose=True,
                handle_parsing_errors=True
            )
    
    async def process(
        self, 
        source_json: Union[Dict[str, Any], List[Any]], 
        target_json: Optional[Union[Dict[str, Any], List[Any]]] = None,
        context: Optional[str] = None
    ) -> AgentResult:
        """
        Generate a JOLT transformation specification from source and target JSON.
        
        This is the main processing method that analyzes the provided JSON structures
        and generates a complete JOLT specification to transform the source into the
        target format. Uses structured output to ensure consistent results and
        provides detailed explanations of the transformation logic.
        
        The method handles both scenarios:
        1. Source + Target: Direct transformation specification generation
        2. Source + Context: Target JSON creation followed by specification generation
        
        Args:
            source_json (Union[Dict, List]): Input JSON data that needs transformation
            target_json (Optional[Union[Dict, List]]): Desired output format example
                                                     (optional if context is provided)
            context (Optional[str]): Additional instructions for transformation logic
                                   or target structure requirements
            
        Returns:
            AgentResult: Result containing either:
                - success=True, result=TransformationResult with JOLT spec and explanation
                - success=False, error=str with detailed error information
                
        Raises:
            Handles all exceptions gracefully and returns them as AgentResult errors
            
        Side Effects:
            - Formats JSON inputs for optimal LLM processing
            - Logs processing details and rich output for debugging
            - Uses structured output validation for result consistency
        """
        # Format source JSON for optimal LLM processing
        if type(source_json) is dict:
            formatted_source = json.dumps(source_json, indent=2)
        else: 
            formatted_source = source_json

        # Format target JSON if provided, otherwise use empty string
        if target_json:
            if type(target_json) is dict:
                formatted_target = json.dumps(target_json, indent=2)
            else: 
                formatted_target = target_json
        else: 
            formatted_target = ''
        
        # Construct comprehensive input prompt for the LLM
        input_text = f"""
        Please create a JOLT transformation specification that transforms the SOURCE JSON into the TARGET JSON. \n Please do not create a JOLT spec that will only work for this input. The JOLT spec you create should work for all input JSON files that come with the same structure. 
        
        SOURCE JSON:
        ```json
        {formatted_source}
        ```
        
        TARGET JSON:
        ```json
        {formatted_target}
        ```
        
        {f'CONTEXT: {context}' if context else ''}
        
        Provide your JOLT specification as a valid JSON array, and include a detailed explanation of how it works.

        """
        
        try:
            from rich import print
            
            if not self.tools:
                # Direct LLM invocation without tools for simpler, faster processing
                logger.info("No tools provided, using direct LLM invocation")
                try:
                    # Prepare messages for structured output generation
                    messages = [
                        SystemMessage(content=self.system_prompt),
                        HumanMessage(content=input_text)
                    ]
                    
                    # Use structured output to ensure consistent result format
                    llm = self.llm.with_structured_output(TransformationResult)
                    print(llm)
                    response = await llm.ainvoke(messages)
                    
                    print(response)

                    return AgentResult(
                        success=True,
                        result=response
                    )
                    
                except Exception as e:
                    logger.error(f"Direct LLM invocation failed: {e}")
                    return AgentResult(
                        success=False,
                        error=f"LLM invocation failed: {str(e)}"
                    )
           
            # Tool-enhanced processing would be implemented here
            # Currently the agent primarily uses direct LLM invocation
            # Future enhancement: Add tool-based validation and refinement
            
        except Exception as e:
            logger.error(f"Error in transformer agent: {e}")
            return AgentResult(
                success=False,
                error=f"Transformation failed: {str(e)}"
            )
    
    def update_tools(self, tools: List[BaseTool]) -> None:
        """
        Update the tools available to the agent and reinitialize the executor.
        
        Allows dynamic reconfiguration of the agent's capabilities by updating
        the tool set. This is useful for adding or removing validation tools,
        execution tools, or other enhancements based on runtime requirements
        or user preferences.
        
        Args:
            tools (List[BaseTool]): New list of LangChain tools to be used by the agent
                                  (replaces existing tools completely)
                                  
        Side Effects:
            - Updates internal tools list
            - Recreates agent executor with new tool configuration
            - Rebinds tools to LLM if tools are provided
            - Logs tool update for debugging and monitoring
        """
        self.tools = tools
        
        # Recreate agent executor with updated tool configuration
        self.agent_executor = self._create_agent_executor()
        logger.info(f"Updated transformer agent with {len(tools)} tools")