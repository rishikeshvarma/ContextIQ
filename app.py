'''import torch
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import google.generativeai as genai
import io
import soundfile as sf

# -----------------------------------------------------
# CONFIG
# -----------------------------------------------------

GEMINI_API_KEY = "gemini_api_key"
WHISPER_MODEL = "openai/whisper-small"

genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="MAX Intelligent System")

# -----------------------------------------------------
# MODELS
# -----------------------------------------------------
# Whisper - local STT
import whisper
whisper_model = whisper.load_model("small")

# -----------------------------------------------------
# INPUT MODEL
# -----------------------------------------------------
class TextIn(BaseModel):
    text: str


# -----------------------------------------------------
# GEMINI NER (super token-efficient)
# -----------------------------------------------------
@app.post("/ner")
def ner(payload: TextIn):

    prompt = f"""
Extract named entities from the text in JSON.
Text: "{payload.text}"

Entities:
- PERSON
- DATE
- DEADLINE
- TASK
- DECISION
- PROJECT
- ORG
- MEETING
- LOCATION
- PRODUCT
- PRIORITY

Return JSON ONLY with:
{{
    "entities":[{{"text":"...", "label":"..."}}]
}}
    """

    response = genai.GenerativeModel("gemini-2.5-flash").generate_content(prompt)
    return response.text


# -----------------------------------------------------
# CLASSIFIER
# -----------------------------------------------------
@app.post("/classify")
def classify(payload: TextIn):

    prompt = f"""
Classify the type of message in ONE WORD.
Labels: STATEMENT, TASK, DECISION, QUESTION, MEETING, UPDATE, DEADLINE, OTHER.
Text: "{payload.text}"

Return JSON:
{{"label":"..."}}"""

    response = genai.GenerativeModel("gemini-2.5-flash").generate_content(prompt)
    return response.text


# -----------------------------------------------------
# EXTRACTION (Decision + Task)
# -----------------------------------------------------
@app.post("/extract")
def extract(payload: TextIn):

    prompt = f"""
Extract structured information in JSON.
Text: "{payload.text}"

Return JSON:
{{
  "decisions": [...],
  "tasks": [...],
  "deadlines": [...],
  "owners": [...]
}}
    """

    response = genai.GenerativeModel("gemini-2.5-flash").generate_content(prompt)
    return response.text


# -----------------------------------------------------
# EMBEDDINGS (Gemini)
# -----------------------------------------------------
@app.post("/embed")
def embed(payload: TextIn):

    embedding = genai.embed_content(
        model="models/text-embedding-004",
        content=payload.text
    )

    return {"embedding": embedding['embedding']}


# -----------------------------------------------------
# GENERATE RESPONSE
# -----------------------------------------------------
@app.post("/generate")
def generate(payload: TextIn):

    prompt = f"""
Generate a helpful, short response to the message:
"{payload.text}"
"""

    response = genai.GenerativeModel("gemini-2.5-flash").generate_content(prompt)
    return {"response": response.text}


# -----------------------------------------------------
# CHAT WITH MEMORY (token-efficient)
# -----------------------------------------------------
chat_memory = []

@app.post("/chat")
def chat(payload: TextIn):

    # Save memory locally (no token cost)
    chat_memory.append(payload.text)
    if len(chat_memory) > 20:
        chat_memory.pop(0)

    context_summary = " ".join(chat_memory[-5:])  # last 5 messages only (token saving)

    prompt = f"""
Context from previous messages:
{context_summary}

User: {payload.text}

Respond concisely.
"""

    response = genai.GenerativeModel("gemini-2.5-flash").generate_content(prompt)
    return {"response": response.text}


# -----------------------------------------------------
# TRANSCRIBE (Whisper Small Local)
# -----------------------------------------------------
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):

    audio_bytes = await file.read()
    audio, sr = sf.read(io.BytesIO(audio_bytes))

    result = whisper_model.transcribe(audio)
    return {"text": result["text"]}'''
