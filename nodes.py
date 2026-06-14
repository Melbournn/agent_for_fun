import requests
import io
import os
import uuid
import google.generativeai as genai
from PIL import Image
from langchain_openai import ChatOpenAI
from state import AgentState
from config import Config

badmf = ChatOpenAI(model="gpt-4o")

def generate_image(prompt: str) -> str:
    """Generates an image using Google Gemini (Imagen) or Hugging Face Inference API."""
    
    # Try Google Gemini first if key is available
    if Config.GOOGLE_API_KEY:
        print(f"Attempting Gemini image generation for prompt: {prompt}")
        try:
            genai.configure(api_key=Config.GOOGLE_API_KEY)
            # Note: Imagen 3 might require specific model names like 'imagen-3.0-generate-001' 
            # or just 'imagen-3' depending on the region/tier.
            model = genai.ImageGenerationModel("imagen-3")
            response = model.generate_images(prompt = prompt, number_of_images = 1)
            
            if response.images:
                os.makedirs("output", exist_ok = True)
                file_path = f"output/{uuid.uuid4()}.png"
                response.images[0].save(file_path)
                return file_path
        except Exception as e:
            print(f"Gemini Image Generation failed: {e}. Falling back to HF...")

    # Fallback to Hugging Face
    hf_token = Config.HF_TOKEN
    model_id = "black-forest-labs/FLUX.1-schnell"
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    
    print(f"Generating image via HF for prompt: {prompt}")
    try:
        response = requests.post(api_url, headers = headers, json = {"inputs": prompt})
        
        if response.status_code == 200:
            image_bytes = response.content
            image = Image.open(io.BytesIO(image_bytes))
            
            os.makedirs("output", exist_ok=True)
            file_path = f"output/{uuid.uuid4()}.png"
            image.save(file_path)
            return file_path
        else:
            print(f"HF Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"HF Image Generation failed: {e}")
        
    return ""

def transcriber(state: AgentState):
    """Converts raw speech input into a clean transcription."""
    prompt = (
        f"You are a transcription assistant. Your job is to take raw speech-to-text data and convert it into a coherent text format. "
        f"Preserve all spoken words, even if they are mixed or messy. Do not translate yet, just transcribe clearly.\n\n"
        f"Raw Input: {state['raw_speech']}"
    )
    response = badmf.invoke(prompt)
    return {"transcription": response.content}

def translator(state: AgentState):
    """Translates mixed speech-text from transcriber to plain English."""
    prompt = (
        f"You are a professional translator. You will receive text that may contain a mix of different languages and speech-to-text errors. "
        f"Your task is to translate and normalize this into clear, professional English, ensuring the original intent is perfectly preserved.\n\n"
        f"Transcription: {state['transcription']}"
    )
    response = badmf.invoke(prompt)
    return {"translated_text": response.content}

def architect(state: AgentState):
    """Converts translated text into a structured prompt for the executor."""
    prompt = (
        f"You are an elite Prompt Engineer. Your objective is to take a user's natural language request and transform it into a structured, high-context instruction set for an autonomous Execution Agent.\n\n"
        f"User Request: {state['translated_text']}\n\n"
        "You must output in a structured format containing:\n"
        "1. Core Objective: One sentence defining the primary goal.\n"
        "2. Technical Constraints: Explicit technical requirements, libraries, or formats.\n"
        "3. Step-by-Step Execution Plan: A logical sequence of operations.\n"
        "4. Safety / Edge Case Handling: Identify one potential failure point and how to handle it.\n"
        "5. Image Prompt: A detailed descriptive prompt for image generation if applicable.\n\n"
        "Rules:\n"
        "1. Do not perform the task; only write the instructions to perform it.\n"
        "2. Avoid 'robotic' or generic fillers. Be technical and precise.\n"
        "3. If the user's request is ambiguous, add a 'Clarification Needed' section asking for missing parameters."
    )
    response = badmf.invoke(prompt)
    return {"structured_prompt": {"content": response.content}}

def executor(state: AgentState):
    """Takes the structured prompt from the architect and performs the task."""
    instructions = state['structured_prompt']['content']
    
    # Extract Image Prompt from instructions
    # We'll use GPT to extract just the image prompt part if it exists
    extract_prompt = (
        f"Extract only the 'Image Prompt' from the following instructions. "
        f"If no specific image prompt is found, create a descriptive one based on the Core Objective. "
        f"Return ONLY the image prompt text, nothing else.\n\n"
        f"INSTRUCTIONS:\n{instructions}"
    )
    image_prompt = badmf.invoke(extract_prompt).content.strip()
    
    image_path = generate_image(image_prompt)
    
    if image_path:
        result = f"Image generated successfully: {image_path}"
    else:
        result = "Failed to generate image."
        
    return {"result": result, "image_path": image_path}
