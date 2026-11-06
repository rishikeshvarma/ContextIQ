"""Backend tests. Gemini, Discord, Hugging Face and Supabase are all faked,
so these need no keys and no network."""
import unittest
from unittest import mock

from fastapi.testclient import TestClient

import new_backend


def fake_gemini(text="hello back"):
    model = mock.Mock()
    model.generate_content.return_value = mock.Mock(text=text)
    return mock.patch.object(new_backend.genai, "GenerativeModel", return_value=model)


class BackendTest(unittest.TestCase):
    def setUp(self):
        # Not used as a context manager, so the startup sync thread stays off.
        self.client = TestClient(new_backend.app)
        new_backend.chat_memory.clear()

    def test_health(self):
        self.assertEqual(self.client.get("/health").json(), {"ok": True})

    def test_chat_replies_and_saves(self):
        db = mock.Mock()
        with fake_gemini(), mock.patch.object(new_backend, "supabase", db):
            res = self.client.post("/chat", json={"text": "hi"})
        self.assertEqual(res.json(), {"response": "hello back"})
        db.table.assert_called_with("messages")
        row = db.table.return_value.insert.call_args.args[0]
        self.assertEqual(row["user_message"], "hi")
        self.assertEqual(row["source"], "chat")

    def test_chat_survives_gemini_failure(self):
        with mock.patch.object(new_backend.genai, "GenerativeModel", side_effect=RuntimeError("down")), \
                mock.patch.object(new_backend, "supabase", None):
            res = self.client.post("/chat", json={"text": "hi"})
        self.assertEqual(res.status_code, 200)
        self.assertIn("couldn't generate", res.json()["response"])

    def test_failed_insert_does_not_stop_the_sync(self):
        db = mock.Mock()
        db.table.return_value.upsert.return_value.execute.side_effect = [RuntimeError("bad row"), None]
        with mock.patch.object(new_backend, "supabase", db):
            new_backend._insert_messages_rows("g1", [{"id": "1"}, {"id": "2"}])
        self.assertEqual(db.table.return_value.upsert.call_count, 2)

    def test_repeat_sync_upserts_on_message_id(self):
        db = mock.Mock()
        with mock.patch.object(new_backend, "supabase", db):
            new_backend._insert_messages_rows("g1", [{"id": "1", "content": "x"}])
        self.assertEqual(db.table.return_value.upsert.call_args.kwargs["on_conflict"], "message_id")

    def test_chat_prompt_includes_saved_discord_messages(self):
        db = mock.MagicMock()
        query = db.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value
        query.execute.return_value = mock.Mock(data=[
            {"channel_name": "dev", "author_username": "bo", "content": "newest"},
            {"channel_name": "dev", "author_username": "ana", "content": "launch is friday"},
        ])
        with fake_gemini() as model, mock.patch.object(new_backend, "supabase", db):
            self.client.post("/chat", json={"text": "when is launch?"})
        prompt = model.return_value.generate_content.call_args.args[0]
        self.assertIn("#dev ana: launch is friday", prompt)
        self.assertLess(prompt.index("launch is friday"), prompt.index("newest"))

    def test_chat_still_answers_when_discord_read_fails(self):
        db = mock.MagicMock()
        db.table.return_value.select.side_effect = RuntimeError("db down")
        with fake_gemini(), mock.patch.object(new_backend, "supabase", db):
            res = self.client.post("/chat", json={"text": "hi"})
        self.assertEqual(res.json(), {"response": "hello back"})

    def test_sync_skips_a_channel_it_cannot_read(self):
        def fake_get(url, headers=None):
            if url.endswith("/channels"):
                return mock.Mock(json=lambda: [{"id": "a", "name": "open", "type": 0},
                                               {"id": "b", "name": "locked", "type": 0},
                                               {"id": "c", "name": "voice", "type": 2}])
            if "/channels/a/" in url:
                return mock.Mock(json=lambda: [{"id": "m1", "content": "ship it"}])
            return mock.Mock(json=lambda: {"message": "Missing Access"})

        with mock.patch.object(new_backend.requests, "get", side_effect=fake_get), \
                mock.patch.object(new_backend, "_insert_messages_rows") as insert:
            new_backend._sync_guild_messages("g1")
        guild, rows = insert.call_args.args
        self.assertEqual(guild, "g1")
        self.assertEqual([(r["id"], r["channel_name"]) for r in rows], [("m1", "open")])

    def test_transcribe_uses_hugging_face(self):
        with mock.patch.object(new_backend.requests, "post",
                               return_value=mock.Mock(json=lambda: {"text": "spoken"})) as post:
            res = self.client.post("/transcribe", files={"file": ("a.wav", b"audio")})
        self.assertEqual(res.json(), {"text": "spoken"})
        self.assertEqual(post.call_args.kwargs["data"], b"audio")

    def test_transcribe_upload_says_when_nothing_was_stored(self):
        with mock.patch.object(new_backend.requests, "post",
                               return_value=mock.Mock(json=lambda: {"text": "spoken"})), \
                mock.patch.object(new_backend, "supabase", None):
            res = self.client.post("/transcribe_upload", files={"file": ("a.wav", b"audio")})
        self.assertEqual(res.json()["status"], "not stored")


if __name__ == "__main__":
    unittest.main()
