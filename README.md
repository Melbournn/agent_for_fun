# Agent for Fun: Audio-to-Visual Agent

An autonomous AI agent that converts speech into high-quality imagery using a structured LangGraph workflow. It transcribes audio, translates mixed-language speech, designs professional image prompts, and executes image generation using Google Gemini or Hugging Face.

## Architecture

The project uses **LangGraph** to manage a stateful workflow:
1.  **Transcriber**: Converts raw audio to text (OpenAI Whisper).
2.  **Translator**: Normalizes transcription into professional English (GPT-4o).
3.  **Architect**: Designs a high-context technical execution plan and a detailed image prompt (GPT-4o).
4.  **Executor**: Generates and saves the final image (Gemini Imagen 3 / FLUX.1).

## Setup

### 1. Prerequisites
*   Python 3.10+
*   OpenAI API Key
*   Google AI (Gemini) API Key (Optional, for Imagen 3)
*   Hugging Face Token (Optional, for FLUX fallback)

### 2. Installation
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_gemini_key
HF_TOKEN=your_huggingface_token
```

## Usage

1.  Place your audio file (mp3, wav, m4a) into the `uploads/` folder.
2.  Run the agent:
```bash
python main.py uploads/your_audio_file.wav
```
3.  Find your generated image in the `output/` folder.

## Project Structure
*   `main.py`: Entry point and LangGraph orchestration.
*   `nodes.py`: Individual agent logic (Transcribe, Translate, Architect, Executor).
*   `receiver.py`: Audio processing and initial transcription.
*   `state.py`: Shared state schema for the agent.
*   `config.py`: Environment variable management.
*   `uploads/`: Directory for input audio files.
*   `output/`: Directory for generated PNG images.

## License
MIT
