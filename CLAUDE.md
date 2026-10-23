# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status
SnackStack is a CLI multi-agent food-delivery assistant built with LangGraph. It is an assignment, and the repo is currently a `uv` package skeleton. The spec is in `resources/Multimodal Conversational System_Assignment.md` (read it before implementing anything). It is course material, so it's gitignored and kept local only; never commit it. The target graph is drawn in `resources/order_router.drawio.svg`.

## Tooling
- Package/env manager: **uv** (build backend `uv_build`, src layout: code lives in `src/snackstack/`). Python 3.12 (`.python-version`).
- Add deps: `uv add <pkg>`. Deps: `langgraph`, `langchain`, `langchain-openai`, `langchain-chroma`/`chromadb`, `pydantic`, `python-dotenv`, `openai`, and for voice `sounddevice` + `numpy` (WAV via stdlib `wave`, no `soundfile`).
- Run: `uv run snackstack` (entry point `snackstack.main:main`). Flags: `--voice` (voice in + out), `--voice-out` (text in, voice out), none for text-only. Voice boundary smoke tests: `uv run snackstack-voice {record,vad,stt,tts,tts-stream}`.
- Tests: `uv run pytest` (offline: tools, VAD, WAV, speech normalization), or `uv run pytest tests/test_voice.py::test_rms` for one.
- Needs `OPENAI_API_KEY` in the environment or `.env`. Ship a `.env.example`.

## Target architecture (from the brief + diagram)
Flow: user input (text or Whisper STT) -> **orchestrator** -> `Send()` fan-out to **menu_agent** and/or **order_agent** -> **synthesizer** -> END -> output (text or OpenAI TTS).

- **State** (`state.py`): `StackState` TypedDict with `messages`, `user_query`, `route`, `menu_response`, `order_response`, `final_answer`. Parallel branches write different keys (`menu_response` and `order_response`), so they don't conflict.
- **Orchestrator**: `llm.with_structured_output(PydanticModel)` returns a list of agent names plus reasoning. Greetings and unclear intent go to `menu_agent`. A query that spans both topics goes to both via `Command` + `Send()`. The prompt must stop greetings from routing to both agents.
- **Menu agent**: tools bound with `llm.bind_tools([search_menu_catalog])`, and a manual tool-calling loop capped at `MAX_TOOL_ITERATIONS` (5) with a fallback answer. Ends with `Command(goto="synthesizer_node")`. The tool queries an in-memory ChromaDB store built at startup from `data/menu.py` (8 dishes) with `text-embedding-3-small`.
- **Order agent**: same loop pattern with `get_order_status`, which looks up by order ID, tracking ID, or email in `data/orders.py` (5 orders keyed by order_id). It first extracts an identifier with regex (`ORD-\d+`, `SS\d+TRK`, email). If none is found, it calls `interrupt()` to ask the user.
- **Synthesizer**: an LLM merges one or both agent outputs into `final_answer`.
- **Graph** (`graph.py`): `StateGraph(StackState)`, START -> orchestrator, synthesizer -> END, compiled with the `MemorySaver` checkpointer. HITL needs a checkpointer, and every invoke needs `config={"configurable": {"thread_id": ...}}`.
- **CLI** (`main.py`): a `SnackStackAssistant` class wraps the graph. `ask()` invokes it, checks `graph.get_state(config)` for pending interrupts, prompts with `input()`, and resumes with `Command(resume=answer)`. The REPL supports `reset` (new thread_id) and `quit`.
- Config (`config.py`) builds `ChatOpenAI(model="gpt-4o")`, `OpenAIEmbeddings`, and the OpenAI client once. Prompts live in `agents/prompts.py`.

Suggested module layout (from the brief, not mandatory): `config.py`, `state.py`, `graph.py`, `logger.py`, `main.py`, `agents/{prompts,orchestrator,menu_agent,order_agent,synthesizer}.py`, `data/{menu,orders}.py`, `tools/{rag,menu_tools,order_tools}.py`, `voice/{recorder,speaker}.py`.

## Voice (`voice.py`)
Framework-free cascade adapted from the instructor's `axiomcart-ai-assistant` repo, branch `workshop/voice-cascade`, file `workshop/local/voice.py`. It goes mic, then energy VAD (`SilenceDetector`/`record_until_silence`), then batch STT (`transcribe`, with a vocabulary prompt), then the graph as the "think" step, then `normalize_for_speech`, then streaming PCM TTS (`speak_stream`). The loop is half-duplex: there's no barge-in. In voice mode the HITL interrupt question is spoken, and the answer is heard through the mic. Default models are `whisper-1`/`tts-1`, because this OpenAI project has no access to `gpt-4o-mini-transcribe`/`gpt-4o-mini-tts`.

## Gotchas from the brief
- Use package-qualified imports (`from snackstack.config import llm`) to avoid circular imports.
- Request **wav** (batch) or **pcm** (streaming, 24 kHz mono int16) from TTS, not mp3. Never play the PCM at the mic's 16 kHz rate.
- ChromaDB warnings on first run are expected, because the store is rebuilt on every start.

## Acceptance scenarios
"hi" -> menu agent greeting. "Show me vegan dishes" -> calls search_menu_catalog. "Track order ORD-201" -> Butter Chicken, Out for Delivery. "Where is my order?" -> interrupt, then lookup. "Check order by email priya@example.com" -> ORD-201. "Pizza options + track ORD-203" -> both agents, merged. "reset" -> new thread. The full table is in section 8 of the brief.
