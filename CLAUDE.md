# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status
SnackStack is a CLI multi-agent food-delivery assistant built with LangGraph, with optional voice in and out. It is an assignment and is fully working: tools, agents, graph, CLI and voice. The spec is in `resources/Multimodal Conversational System_Assignment.md`. It is course material, so it's gitignored and kept local only; never commit it. The architecture diagram is `resources/architecture.svg` (shown in the README). It's a hand-written SVG in the brand palette; keep it in sync when the graph changes.

## Code style
Keep Python simple and readable for an intermediate developer. Prefer plain loops, `in` checks and small functions over regex, clever one-liners or extra layers. Add complexity only when a concrete need calls for it, and say why in a short comment.

## Tooling
- Package/env manager: **uv** (build backend `uv_build`, src layout: code lives in `src/snackstack/`). Python 3.12 (`.python-version`).
- Add deps: `uv add <pkg>`. Deps: `langgraph`, `langchain`, `langchain-openai`, `langchain-chroma`/`chromadb`, `pydantic`, `python-dotenv`, `openai`, and for voice `sounddevice` + `numpy` (WAV via stdlib `wave`, no `soundfile`).
- Run: `uv run snackstack` (entry point `snackstack.main:main`). Flags: `--voice` (voice in + out), `--voice-out` (text in, voice out), none for text-only. Voice boundary smoke tests: `uv run snackstack-voice {record,vad,stt,tts,tts-stream}`.
- Tests: `uv run pytest` (offline: order tools, VAD, WAV, speech normalization), or `uv run pytest tests/test_voice.py::test_rms` for one.
- Needs `OPENAI_API_KEY` in the environment or `.env`. `.env.example` is the template.

## Architecture
Flow: user input (text, or mic -> VAD -> Whisper STT) -> **orchestrator** -> `Send()` fan-out to **menu_agent** and/or **order_agent** -> **synthesizer** -> END -> output (text, or streaming TTS).

- **State** (`state.py`): `StackState` TypedDict with `messages`, `user_query`, `route`, `menu_response`, `order_response`, `final_answer`. Parallel branches write different keys (`menu_response` and `order_response`), so they don't conflict.
- **Orchestrator** (`agents/orchestrator.py`): `llm.with_structured_output(RouteDecision)` returns a list of agent names plus reasoning. Greetings and unclear intent go to `menu_agent`; a query that spans both topics goes to both via `Command` + `Send()`. It clears the previous turn's agent outputs.
- **Tool loop** (`agents/_loop.py`): `run_tool_loop()` is shared by both agents. It binds the tools, runs the LLM -> tool -> LLM loop, and stops at `MAX_TOOL_ITERATIONS` (5) with a fallback answer.
- **Menu agent**: uses `search_menu_catalog`, which queries an in-memory ChromaDB store built at startup from `data/menu.json` (8 dishes) with `text-embedding-3-small` (`tools/rag.py`, k=6). Ends with `Command(goto="synthesizer_node")`.
- **Order agent**: uses `get_order_status`, one plain function that looks up an order ID, tracking ID or email in `data/orders.json` (5 orders keyed by order_id). `has_identifier()` checks the message for `ORD-`, `TRK` or `@` with plain substring tests (no regex, by design). If none is found, it calls `interrupt()` before any LLM call, because the node re-runs from the top on resume. The LLM pulls the actual ID out of the message when it calls the tool.
- **Synthesizer**: an LLM merges one or both agent outputs into `final_answer` and appends it to `messages`.
- **Graph** (`graph.py`): `StateGraph(StackState)`, START -> orchestrator, synthesizer -> END, compiled with the `MemorySaver` checkpointer. HITL needs a checkpointer, and every invoke needs `config={"configurable": {"thread_id": ...}}`.
- **CLI** (`main.py`): `SnackStackAssistant` wraps the graph. `ask(text, ask_user)` invokes it, loops while the result has `__interrupt__`, gets an answer from `ask_user` (`input()` in text mode, speak-and-listen in voice mode), and resumes with `Command(resume=answer)`. The REPL supports `reset` (new thread_id) and `quit`.
- Config (`config.py`) loads `.env` and builds `ChatOpenAI` (default `gpt-4o`), `OpenAIEmbeddings`, and the OpenAI client once. Prompts live in `agents/prompts.py`.

## Voice (`voice.py`)
Framework-free cascade adapted from the instructor's `axiomcart-ai-assistant` repo, branch `workshop/voice-cascade`, file `workshop/local/voice.py`. It goes mic, then energy VAD (`SilenceDetector`/`record_until_silence`), then batch STT (`transcribe`, with a vocabulary prompt), then the graph as the "think" step, then `normalize_for_speech` (plain string steps: drop markdown, "Rs" -> "rupees", spell out IDs), then streaming PCM TTS (`speak_stream`). The loop is half-duplex: there's no barge-in. In voice mode the HITL interrupt question is spoken, and the answer is heard through the mic. Default models are `whisper-1`/`tts-1`, because this OpenAI project has no access to `gpt-4o-mini-transcribe`/`gpt-4o-mini-tts`.

## Gotchas
- Use package-qualified imports (`from snackstack.config import llm`) to avoid circular imports.
- Request **wav** (batch) or **pcm** (streaming, 24 kHz mono int16) from TTS, not mp3. Never play the PCM at the mic's 16 kHz rate.
- ChromaDB warnings on first run are expected, because the store is rebuilt on every start.

## Acceptance scenarios
"hi" -> menu agent greeting. "Show me vegan dishes" -> calls search_menu_catalog. "Track order ORD-201" -> Butter Chicken, Out for Delivery. "Where is my order?" -> interrupt, then lookup. "Check order by email priya@example.com" -> ORD-201. "Pizza options + track ORD-203" -> both agents, merged. "reset" -> new thread. The full table is in section 8 of the brief.
