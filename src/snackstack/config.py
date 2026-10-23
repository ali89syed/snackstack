"""LLM, embeddings, and OpenAI client setup. Built once and shared."""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
    )

CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
MAX_TOOL_ITERATIONS = 5

# Defaults work on any OpenAI project. If yours has access, gpt-4o-mini-transcribe /
# gpt-4o-mini-tts (voice "marin") are faster and support a speaking-style prompt.
VOICE_STT_MODEL = os.getenv("VOICE_STT_MODEL", "whisper-1")
VOICE_TTS_MODEL = os.getenv("VOICE_TTS_MODEL", "tts-1")
VOICE_TTS_VOICE = os.getenv("VOICE_TTS_VOICE", "alloy")

llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
openai_client = OpenAI()
