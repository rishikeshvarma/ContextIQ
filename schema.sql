-- Run once in the Supabase SQL editor. Reconstructed from what new_backend.py
-- reads and writes, since the original schema was never committed.

create table if not exists messages (
    id              bigint generated always as identity primary key,
    created_at      timestamptz not null default now(),
    source          text not null,          -- 'chat' or 'discord'
    -- /chat rows
    user_message    text,
    bot_response    text,
    -- Discord rows
    guild_id        text,
    channel_id      text,
    channel_name    text,
    author_id       text,
    author_username text,
    message_id      text unique,            -- null on chat rows; stops repeat syncs saving duplicates
    content         text,
    raw             jsonb
);
create index if not exists messages_source_created_at on messages (source, created_at desc);

create table if not exists audio_transcripts (
    id         bigint generated always as identity primary key,
    created_at timestamptz not null default now(),
    transcript text not null
);

create table if not exists discord_users (
    id                bigint generated always as identity primary key,
    created_at        timestamptz not null default now(),
    user_access_token text,
    guilds            jsonb,
    selected_guild    text
);

create table if not exists discord_events (
    id         bigint generated always as identity primary key,
    created_at timestamptz not null default now(),
    event      jsonb not null
);

-- These tables hold Discord tokens and private messages. Row level security with
-- no policies blocks the public anon key; the service role key still works.
alter table messages enable row level security;
alter table audio_transcripts enable row level security;
alter table discord_users enable row level security;
alter table discord_events enable row level security;
