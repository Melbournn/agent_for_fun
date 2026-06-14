import os
from openai import OpenAI
from dotenv import load_dotenv
from state import AgentState

load_dotenv()
client = OpenAI()

def audio_receiver(audio_file_path: str) -> AgentState:
    """
    Takes an audio file path, transcribes it using Whisper, 
    and returns the initial AgentState.
    """
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

    with open(audio_file_path, "rb") as audio_file:
        # Using Whisper-1 model via OpenAI API
        transcription = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file,
            response_format="text"
        )
    
    # Initializing the state with the raw transcription from Whisper
    state: AgentState = {
        "raw_speech": transcription,
        "transcription": "",
        "translated_text": "",
        "structured_prompt": {},
        "result": ""
    }
    
    return state
