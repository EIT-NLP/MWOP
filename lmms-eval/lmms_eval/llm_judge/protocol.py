from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
DEFAULT_NUM_RETRIES = 5
DEFAULT_RETRY_DELAY = 10

@dataclass
class ServerConfig:
    model_name: str
    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: Optional[float] = None
    timeout: int = 60
    num_retries: int = DEFAULT_NUM_RETRIES
    retry_delay: float = DEFAULT_RETRY_DELAY
    max_concurrent: int = 10
    system_prompt: Optional[str] = None
    response_format: Optional[str] = None
    judge_type: str = 'general'
    output_format: Optional[str] = None
    score_range: Optional[Tuple[float, float]] = None
    evaluation_criteria: Optional[Dict[str, Any]] = None

@dataclass
class Request:
    messages: List[Dict[str, Any]]
    images: Optional[List[Union[str, bytes]]] = None
    config: Optional[ServerConfig] = None
    question: Optional[str] = None
    answer: Optional[str] = None
    prediction: Optional[str] = None
    context: Optional[str] = None
    options: Optional[List[str]] = None
    response1: Optional[str] = None
    response2: Optional[str] = None
    custom_prompt: Optional[str] = None
    prompt_kwargs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Response:
    content: str
    model_used: str
    usage: Optional[Dict[str, int]] = None
    raw_response: Optional[Any] = None
    parsed_result: Optional[Union[int, float, bool, Tuple[float, float], Dict[str, Any]]] = None
    success: bool = True
    error_message: Optional[str] = None
