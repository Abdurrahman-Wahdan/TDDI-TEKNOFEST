from typing import TypedDict, Dict, Any, List, Optional

class WorkflowState(TypedDict):
    user_input: str
    assistant_response: str
    last_assistant_response: str
    required_user_input: str
    required_response: bool
    agent_message: str
    customer_id: str
    tool_group: str                  # From classifier's json_output.tool
    operation_in_progress: bool       # Track if operation is ongoing
    available_tools: List[Dict]       # Tools available for current group
    selected_tool: str               # Currently selected tool
    tool_params: Dict[str, Any]      # Collected parameters
    missing_params: List[str]        # Parameters still needed
    important_data: Dict[str, Any]
    current_process: str
    in_process: str
    chat_summary: str
    chat_history: List[Dict[str, Any]]
    error : str
    json_output: Dict[str, Any]
    last_mcp_output: Dict[str, Any]
    current_tool: str
    current_category: str