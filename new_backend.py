import os
import io
import time
import threading
import logging
from typing import List, Optional

import torch
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
import requests
import google.generativeai as genai
import soundfile as sf
from dotenv import load_dotenv

# -----------------------------------------------------
# LOAD ENV
# -----------------------------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

DISCORD_CLIENT_ID = os.getenv("DISCORD_CLIENT_ID")
DISCORD_CLIENT_SECRET = os.getenv("DISCORD_CLIENT_SECRET")
DISCORD_REDIRECT_URI = os.getenv("DISCORD_REDIRECT_URI")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend")

# -----------------------------------------------------
# CONFIGURE MODELS
# -----------------------------------------------------
try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    logger.warning("Could not configure Gemini: %s", e)

# whisper (local)
import whisper
try:
    whisper_model = whisper.load_model(WHISPER_MODEL)
except Exception as e:
    logger.warning("Whisper model load failed: %s", e)
    whisper_model = None

# -----------------------------------------------------
# FASTAPI
# -----------------------------------------------------
app = FastAPI(title="MAX Intelligent System")

# -----------------------------------------------------
# SUPABASE CLIENT
# -----------------------------------------------------
from supabase import create_client, Client
supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        logger.warning("Could not create supabase client: %s", e)
else:
    logger.warning("Supabase env not found; DB routes will error until configured")

# -----------------------------------------------------
# MODELS / SCHEMAS
# -----------------------------------------------------
class TextIn(BaseModel):
    text: str

# -----------------------------------------------------
# SIMPLE GEMINI ROUTES
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
    try:
        response = genai.GenerativeModel("gemini-2.5-flash").generate_content(prompt)
        return response.text
    except Exception as e:
        logger.exception("NER error")
        return {"error": str(e)}


@app.post("/classify")
def classify(payload: TextIn):
    prompt = f"""
Classify the type of message in ONE WORD.
Labels: STATEMENT, TASK, DECISION, QUESTION, MEETING, UPDATE, DEADLINE, OTHER.
Text: "{payload.text}"

Return JSON:
{{"label":"..."}}"""
    try:
        response = genai.GenerativeModel("gemini-2.5-flash").generate_content(prompt)
        return response.text
    except Exception as e:
        logger.exception("Classify error")
        return {"error": str(e)}


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
    try:
        response = genai.GenerativeModel("gemini-2.5-flash").generate_content(prompt)
        return response.text
    except Exception as e:
        logger.exception("Extract error")
        return {"error": str(e)}


@app.post("/embed")
def embed(payload: TextIn):
    try:
        embedding = genai.embed_content(
            model="models/text-embedding-004",
            content=payload.text
        )
        return {"embedding": embedding['embedding']}
    except Exception as e:
        logger.exception("Embedding error")
        return {"error": str(e)}


@app.post("/generate")
def generate(payload: TextIn):
    prompt = f"""
Generate a helpful, short response to the message:
"{payload.text}"
"""
    try:
        response = genai.GenerativeModel("gemini-2.5-flash").generate_content(prompt)
        return {"response": response.text}
    except Exception as e:
        logger.exception("Generate error")
        return {"error": str(e)}

# -----------------------------------------------------
# CHAT (in-memory) + DB
# -----------------------------------------------------
chat_memory: List[str] = []

@app.post("/chat")
def chat(payload: TextIn):
    # save locally and respond
    chat_memory.append(payload.text)
    if len(chat_memory) > 50:
        chat_memory.pop(0)
    context_summary = " ".join(chat_memory[-8:])
    prompt = f"""
Context from previous messages:
{context_summary}

User: {payload.text}

Respond concisely.
"""
    try:
        response = genai.GenerativeModel("gemini-2.5-flash").generate_content(prompt)
        reply = response.text
    except Exception as e:
        logger.exception("Chat error")
        reply = "Sorry, I couldn't generate a response right now."

    # store minimal chat history to supabase if available
    if supabase:
        try:
            supabase.table("messages").insert({
                "user_message": payload.text,
                "bot_response": reply,
                "source": "chat"
            }).execute()
        except Exception as e:
            logger.warning("Failed to save chat to supabase: %s", e)

    return {"response": reply}


@app.post("/chat_db")
def chat_db(payload: TextIn):
    # same as /chat but explicit DB save
    return chat(payload)

# -----------------------------------------------------
# TRANSCRIBE (HuggingFace Whisper-small API) + upload
# -----------------------------------------------------
HF_API_KEY = os.getenv("HF_API_KEY")
HF_WHISPER_MODEL = os.getenv("HF_WHISPER_MODEL", "https://api-inference.huggingface.co/models/openai/whisper-small")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    res = requests.post(HF_WHISPER_MODEL, headers=headers, data=audio_bytes)
    try:
        text = res.json().get("text")
    except:
        text = None
    return {"text": text}

@app.post("/transcribe_upload")
async def transcribe_and_store(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    res = requests.post(HF_WHISPER_MODEL, headers=headers, data=audio_bytes)
    try:
        text = res.json().get("text")
    except:
        text = None

    if supabase and text:
        try:
            supabase.table("audio_transcripts").insert({"transcript": text}).execute()
        except Exception as e:
            logger.warning("Failed to store transcript: %s", e)

    return {"text": text, "status": "stored in db"}
# -----------------------------------------------------
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    audio, sr = sf.read(io.BytesIO(audio_bytes))
    if whisper_model is None:
        return {"error": "Whisper model not available"}
    result = whisper_model.transcribe(audio)
    return {"text": result.get("text")}


@app.post("/transcribe_upload")
async def transcribe_and_store(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    audio, sr = sf.read(io.BytesIO(audio_bytes))
    if whisper_model is None:
        return {"error": "Whisper model not available"}
    result = whisper_model.transcribe(audio)
    text = result.get("text")
    if supabase:
        try:
            supabase.table("audio_transcripts").insert({"transcript": text}).execute()
        except Exception as e:
            logger.warning("Failed to store transcript: %s", e)
    return {"text": text, "status": "stored in db"}

# -----------------------------------------------------
# DISCORD OAUTH + BOT FLOW
# -----------------------------------------------------

# discord oauth url for user login
DISCORD_OAUTH_URL = (
    "https://discord.com/api/oauth2/authorize"
    f"?client_id={DISCORD_CLIENT_ID}"
    "&response_type=code"
    "&scope=identify%20guilds"
    f"&redirect_uri={DISCORD_REDIRECT_URI}"
)

@app.get("/discord/login")
def discord_login():
    return RedirectResponse(DISCORD_OAUTH_URL)


@app.get("/oauth/discord/callback")
def discord_callback(code: str):
    # exchange code for user token
    token_res = requests.post(
        "https://discord.com/api/oauth2/token",
        data={
            "client_id": DISCORD_CLIENT_ID,
            "client_secret": DISCORD_CLIENT_SECRET,
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": DISCORD_REDIRECT_URI,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    ).json()

    access_token = token_res.get("access_token")
    if not access_token:
        logger.error("No access_token in token_res: %s", token_res)
        return RedirectResponse("/discord/no_admin_guild")

    # fetch user's guilds
    guilds = requests.get(
        "https://discord.com/api/v10/users/@me/guilds",
        headers={"Authorization": f"Bearer {access_token}"}
    ).json()

    # Option B: select guilds where user has ADMIN permission
    admin_guilds = [g for g in (guilds or []) if (int(g.get("permissions", 0)) & 0x8) != 0]
    if not admin_guilds:
        return RedirectResponse("/discord/no_admin_guild")

    selected = admin_guilds[0]
    selected_guild_id = selected.get("id")

    # save user info + selected guild
    if supabase:
        try:
            supabase.table("discord_users").insert({
                "user_access_token": access_token,
                "guilds": guilds,
                "selected_guild": selected_guild_id
            }).execute()
        except Exception as e:
            logger.warning("Failed to save discord_user: %s", e)

    # redirect to bot invite (preselect guild)
    bot_auth_url = (
        "https://discord.com/api/oauth2/authorize"
        f"?client_id={DISCORD_CLIENT_ID}"
        "&scope=bot%20applications.commands"
        f"&guild_id={selected_guild_id}"
        "&permissions=8"
        f"&redirect_uri={DISCORD_REDIRECT_URI}"
    )
    return RedirectResponse(bot_auth_url)


@app.get("/discord/guild/{guild_id}/authorize_bot")
def authorize_bot(guild_id: str):
    bot_oauth = (
        "https://discord.com/api/oauth2/authorize"
        f"?client_id={DISCORD_CLIENT_ID}"
        "&scope=bot%20applications.commands"
        f"&guild_id={guild_id}"
        "&permissions=8"
        f"&redirect_uri={DISCORD_REDIRECT_URI}"
    )
    return RedirectResponse(bot_oauth)


@app.get("/discord/guild/{guild_id}/channels")
def list_guild_channels(guild_id: str):
    headers = {"Authorization": f"Bot {DISCORD_BOT_TOKEN}"}
    res = requests.get(f"https://discord.com/api/v10/guilds/{guild_id}/channels", headers=headers)
    try:
        channels = res.json()
    except Exception as e:
        logger.warning("List channels json error: %s", e)
        return {"error": str(e)}
    return {"channels": channels}


@app.get("/discord/permissions/{guild_id}")
def permissions_check(guild_id: str):
    """Return a per-channel permission check for the bot."""
    headers = {"Authorization": f"Bot {DISCORD_BOT_TOKEN}"}
    channels = requests.get(f"https://discord.com/api/v10/guilds/{guild_id}/channels", headers=headers).json()
    report = []
    for ch in channels:
        if not isinstance(ch, dict):
            continue
        ch_id = ch.get("id")
        # fetch permission overwrite for the bot role via API is not possible easily; try fetching channel info
        res = requests.get(f"https://discord.com/api/v10/channels/{ch_id}", headers=headers)
        try:
            ch_info = res.json()
        except:
            ch_info = {"error": "invalid json"}
        report.append({"channel": ch.get("name"), "id": ch_id, "raw": ch_info})
    return {"report": report}


# -----------------------------------------------------
# SYNC HELPERS (per-message insertion)
# -----------------------------------------------------

def _insert_messages_rows(guild_id: str, all_messages: List[dict]):
    if not supabase:
        logger.warning("Supabase not configured - skipping insert")
        return

    for msg in all_messages:
        try:
            author = msg.get("author", {}) or {}

            supabase.table("messages").insert({
                "source": "discord",
                "guild_id": guild_id,
                "channel_id": msg.get("channel_id"),
                "channel_name": msg.get("channel_name"),
                "author_id": author.get("id"),
                "author_username": author.get("username"),
                "message_id": msg.get("id"),
                "content": msg.get("content"),
                "raw": msg
            }).execute()
        except Exception as e:
            logger.warning("failed insert msg %s: %s", msg.get("id"), e)("failed insert msg %s: %s", msg.get("id"), e)


def _sync_guild_messages(guild_id: str, limit: int = 100):
    headers = {"Authorization": f"Bot {DISCORD_BOT_TOKEN}"}
    res = requests.get(f"https://discord.com/api/v10/guilds/{guild_id}/channels", headers=headers)
    try:
        channels = res.json()
    except Exception as e:
        logger.warning("Could not fetch channels for %s: %s", guild_id, e)
        return

    # show debug
    logger.info("SYNC DEBUG CHANNELS: %s", channels)

    all_messages = []

    for ch in channels:
        if not isinstance(ch, dict):
            continue
        ctype = ch.get("type")
        # fetch from text channels, threads and announcement channels
        if ctype not in [0, 11, 12, 15, 5]:
            continue

        # GET messages - if channel is a thread or forum this returns posts/messages differently
        msgs_res = requests.get(f"https://discord.com/api/v10/channels/{ch['id']}/messages?limit={limit}", headers=headers)
        try:
            msgs = msgs_res.json()
        except Exception as e:
            logger.warning("Invalid JSON for messages in channel %s: %s", ch.get("id"), e)
            continue

        if isinstance(msgs, dict):
            # could be error like Missing Access
            logger.info("Discord returned error for channel %s: %s", ch.get("id"), msgs)
            continue

        if not isinstance(msgs, list):
            logger.info("Unexpected message format in channel %s: %s", ch.get("id"), type(msgs))
            continue

        for m in msgs:
            if not isinstance(m, dict):
                continue
            m["channel_id"] = ch.get("id")
            m["channel_name"] = ch.get("name")
            all_messages.append(m)

    # insert per-row
    _insert_messages_rows(guild_id, all_messages)


@app.get("/discord/guild/{guild_id}/messages")
def get_guild_messages(guild_id: str):
    _sync_guild_messages(guild_id)
    return {"saved": True}


@app.get("/discord/dm_messages")
def get_dm_messages():
    headers = {"Authorization": f"Bot {DISCORD_BOT_TOKEN}"}
    dm_channels = requests.get("https://discord.com/api/v10/users/@me/channels", headers=headers).json()
    all_dms = []
    for dm in dm_channels:
        msgs = requests.get(f"https://discord.com/api/v10/channels/{dm['id']}/messages?limit=50", headers=headers).json()
        for m in msgs:
            m["dm_channel_id"] = dm.get("id")
            all_dms.append(m)
    _insert_messages_rows("dm", all_dms)
    return {"saved": True, "count": len(all_dms)}


@app.post("/discord/events")
def discord_events(payload: dict):
    if not supabase:
        return {"ok": False, "error": "supabase not configured"}
    supabase.table("discord_events").insert({"event": payload}).execute()
    return {"ok": True}


# -------------------------------------------------------------
# PERIODIC SYNC (every 5 minutes) - only sync guilds where user has ADMIN and bot is present
# -------------------------------------------------------------
_scheduler_started = False


def _periodic_sync_loop(interval_seconds: int = 300):
    while True:
        try:
            if not DISCORD_BOT_TOKEN:
                logger.warning("No DISCORD_BOT_TOKEN set, skipping periodic sync")
                time.sleep(interval_seconds)
                continue

            headers = {"Authorization": f"Bot {DISCORD_BOT_TOKEN}"}
            guilds_resp = requests.get("https://discord.com/api/v10/users/@me/guilds", headers=headers).json()

            if isinstance(guilds_resp, dict):
                logger.warning("Error fetching bot guilds: %s", guilds_resp)
                time.sleep(interval_seconds)
                continue

            # keep only guilds where the bot is present and where the selected user is admin (we use stored selected_guilds)
            if supabase:
                try:
                    rows = supabase.table("discord_users").select("selected_guild").execute()
                    selected_ids = [r.get("selected_guild") for r in (rows.data or []) if r.get("selected_guild")]
                except Exception as e:
                    logger.warning("Could not read discord_users: %s", e)
                    selected_ids = []
            else:
                selected_ids = []

            # sync only selected ids to avoid Missing Access
            for gid in selected_ids:
                try:
                    _sync_guild_messages(gid)
                except Exception as exc:
                    logger.warning("Error syncing guild %s: %s", gid, exc)
        except Exception as e:
            logger.exception("Periodic sync top-level error")
        time.sleep(interval_seconds)


@app.on_event("startup")
def _start_periodic_sync():
    global _scheduler_started
    if not _scheduler_started:
        t = threading.Thread(target=_periodic_sync_loop, args=(300,), daemon=True)
        t.start()
        _scheduler_started = True


@app.post("/discord/sync/{guild_id}")
def discord_sync(guild_id: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(_sync_guild_messages, guild_id)
    return {"sync_started": True}


@app.get("/discord/bot_success")
def bot_success():
    # run a once-off sync for the saved selected guild
    try:
        if supabase:
            res = supabase.table("discord_users").select("selected_guild").order("id", desc=True).limit(1).execute()
            if res.data and res.data[0].get("selected_guild"):
                gid = res.data[0].get("selected_guild")
                _sync_guild_messages(gid)
    except Exception as e:
        logger.warning("Auto-sync error: %s", e)
    return HTMLResponse("<h1>Bot Added Successfully 🎉</h1><p>Your bot is now installed and messages are being synced automatically.</p>")


@app.get("/discord/no_admin_guild")
def no_admin():
    return HTMLResponse("<h2>No admin permissions in any servers.</h2>")


@app.get("/discord/connected")
def connected_guilds():
    if not supabase:
        return {"connected": []}
    res = supabase.table("discord_users").select("guilds", "selected_guild").execute()
    return {"connected": res.data}


# -----------------------------------------------------
# SMALL UTIL: health
# -----------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True}


# End of file
