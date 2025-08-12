from typing import TypedDict, Dict, Any, List

class WorkflowState(TypedDict):
    user_input: str
    assistant_response: str
    get_user_input: bool
    important_data: Dict[str, Any]
    current_process: str
    in_process: str
    chat_summary: str
    chat_history: List[Dict[str, Any]]
    error : str
    json_output: Dict[str, Any]