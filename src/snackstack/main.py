"""CLI entry point: text, voice-out, or full voice conversation with SnackStack."""

import argparse
import uuid
from collections.abc import Callable

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from snackstack.graph import graph
from snackstack.logger import setup_logger

logger = setup_logger(__name__)

RESET_WORDS = {"reset", "start over", "new conversation"}
QUIT_WORDS = {"quit", "exit", "bye", "goodbye", "stop"}


class SnackStackAssistant:
    """Wraps the compiled graph and handles human-in-the-loop interrupts."""

    def __init__(self):
        self.graph = graph
        self.reset()

    @property
    def config(self) -> dict:
        return {"configurable": {"thread_id": self.thread_id}}

    def reset(self):
        self.thread_id = str(uuid.uuid4())
        logger.info("New conversation thread %s", self.thread_id)

    def ask(self, text: str, ask_user: Callable[[str], str] = input) -> str:
        """Run one turn. If the graph pauses on interrupt(), get an answer via ask_user and resume."""
        result = self.graph.invoke(
            {"messages": [HumanMessage(text)], "user_query": text}, config=self.config
        )
        while interrupts := result.get("__interrupt__"):
            answer = ask_user(str(interrupts[0].value))
            result = self.graph.invoke(Command(resume=answer), config=self.config)
        return result.get("final_answer") or "Sorry, I didn't get an answer for that."


def _command(text: str) -> str:
    """Normalize typed or transcribed text for reset/quit matching ("Reset." -> "reset")."""
    return text.lower().strip().strip(".!?,")


def run(voice_in: bool, voice_out: bool):
    if voice_in or voice_out:
        from snackstack import voice  # audio device libraries only load when needed

    def say(text: str):
        print(f"\nSnackStack: {text}\n")
        if voice_out:
            try:
                voice.speak_stream(text)
            except KeyboardInterrupt:
                print("(playback stopped)")

    def hear(prompt: str = "You: ") -> str:
        if not voice_in:
            return input(prompt).strip()
        while True:
            try:
                text = voice.listen()
            except voice.NoSpeechError:
                continue  # silent room: keep listening
            if text:
                print(f"{prompt}{text}")
                return text

    def ask_user(question: str) -> str:
        say(question)
        return hear("You: ")

    assistant = SnackStackAssistant()
    mode = "voice" if voice_in else "voice-out" if voice_out else "text"
    print(f"SnackStack ({mode} mode). Say or type 'reset' for a new conversation, 'quit' to exit.")
    say("Hi! I'm SnackStack. Ask me about the menu or track an order.")

    while True:
        try:
            text = hear()
        except (KeyboardInterrupt, EOFError):
            break
        if not text:
            continue
        if _command(text) in QUIT_WORDS:
            break
        if _command(text) in RESET_WORDS:
            assistant.reset()
            say("Starting a fresh conversation. What can I get you?")
            continue
        try:
            say(assistant.ask(text, ask_user=ask_user))
        except KeyboardInterrupt:
            print("(cancelled)")
        except Exception:
            logger.exception("Turn failed")
            say("Sorry, something went wrong. Please try again.")
    print("Bye!")


def main():
    parser = argparse.ArgumentParser(description="SnackStack multi-agent food delivery assistant")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--voice", action="store_true", help="voice in and voice out")
    group.add_argument("--voice-out", action="store_true", help="text in, voice out")
    args = parser.parse_args()
    run(voice_in=args.voice, voice_out=args.voice or args.voice_out)


if __name__ == "__main__":
    main()
