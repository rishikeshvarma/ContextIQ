# ContextIQ

A FastAPI backend that collects a team's Discord messages into Supabase and
uses Gemini to chat, classify messages and pull out tasks, decisions and
deadlines. It also transcribes audio through Gemini, up to 19 MB per file.

## Run it

1. `python -m venv .venv`, activate it, then `pip install -r requirements.txt`
   (Python 3.12).
2. Create a Supabase project and run `schema.sql` in its SQL editor.
3. Create a Discord application with a bot, and add the redirect URI from step 4.
4. `cp .env.example .env` and fill it in.
5. `uvicorn new_backend:app --reload`, then open `/discord/login` to connect a
   server you administer.

## Access

Send your `API_KEY` as the `X-API-Key` header on every request. Only `/health`
and the pages a browser needs to connect Discord are public. The bot asks only
to view channels and read message history.

## Tests

`python -m unittest discover -t . -s tests`

The tests fake every outside service, so they need no keys or network.

## Layout

- `new_backend.py`: the app.
- `app.py`: an older version, entirely commented out and not used.
