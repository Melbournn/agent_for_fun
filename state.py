from typing import TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    raw_speech: str
    transcription: str
    translated_text: str
    structured_prompt: dict
    result: str
    image_path: str
