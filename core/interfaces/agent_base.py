from pydantic import Field,BaseModel
from abc import ABC,abstractmethod
from typing import Dict,Any,Optional,Union,List


class AgentResult(BaseModel):

    success: bool
    result: Optional[Dict[str,Any]] = None
    error : Optional[str] = None


class BaseAgent(ABC):


    """
    Base interface for all agents in the Jolt Transformation Engine.
    This abstract class defines the contract taht all agent implementations must follow,
    enabling a consistent interface for different types of agents
    """

    @abstractmethod

    async def process(self,*args,**kwargs) -> AgentResult:

        """
        Process a request using the agent's capabilities.

        Args: 
            *args: Positional arguments specific to the agent type
            **kwargs: Keyword arguments specific to the agent type
        """
        pass

    
