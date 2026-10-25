"""Voice I/O for SnackStack: record -> VAD -> STT, and TTS -> speaker. No framework.

Adapted from the voice-cascade workshop (axiomcart-ai-assistant, workshop/local/voice.py).
The "think" stage of that cascade is the SnackStack graph (see main.py).

Try each boundary on its own:  uv run snackstack-voice --help
"""

import argparse
import io
import time
import wave

import numpy as np
import sounddevice as sd

from snackstack.config import VOICE_STT_MODEL, VOICE_TTS_MODEL, VOICE_TTS_VOICE, openai_client

RATE = 16000  # mic capture: 16 kHz mono int16 = 32 KB/s
FRAME_MS = 30
CHANNELS = 1
SAMPLE_DTYPE = "int16"
TTS_PCM_RATE = 24000  # OpenAI TTS "pcm" is 24 kHz mono int16 little-endian, no header
PRE_ROLL_MS = 300

# Biases STT toward our identifiers so "ORD-201" isn't heard as "Ord 2 0 1" or "odd 201".
STT_VOCABULARY = (
    "SnackStack food delivery. Order IDs like ORD-201, ORD-203. Tracking IDs like SS201TRK. "
    "Emails like priya@example.com. Dishes: Margherita Pizza, Vegan Pasta Primavera, "
    "Butter Chicken, Vegan Buddha Bowl, Classic Cheeseburger, Paneer Tikka, Aglio e Olio, Mango Lassi."
)
TTS_INSTRUCTIONS = "Speak in a warm, friendly, upbeat tone, at a natural conversational pace."


# --- Capture --------------------------------------------------------------------------

def to_wav_bytes(samples, rate=RATE) -> bytes:
    """Package int16 samples in a WAV container (format metadata + raw PCM)."""
    samples = np.asarray(samples, dtype=np.int16).reshape(-1)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(samples.tobytes())
    return buf.getvalue()


def save_wav(path, samples, rate=RATE):
    with open(path, "wb") as f:
        f.write(to_wav_bytes(samples, rate))


def record_fixed(seconds=5):
    """Simplest baseline: capture exactly N seconds, then stop."""
    print(f"Recording for {seconds:g} seconds...")
    audio = sd.rec(int(seconds * RATE), samplerate=RATE, channels=CHANNELS, dtype=SAMPLE_DTYPE)
    sd.wait()
    return audio.reshape(-1)


# --- VAD ------------------------------------------------------------------------------

def rms(samples):
    """Normalized signal energy; loudness is NOT the same as speech."""
    x = np.asarray(samples, dtype=np.float32) / 32768.0
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


class SilenceDetector:
    """Tiny energy-based turn detector for teaching, not production VAD."""

    def __init__(self, threshold=0.025, silence_ms=700, max_ms=15000):
        self.threshold, self.silence_ms, self.max_ms = threshold, silence_ms, max_ms
        self.elapsed_ms = 0
        self.first_speech_ms = None
        self.last_speech_ms = None

    def update(self, volume, frame_ms=FRAME_MS):
        """Feed one frame's volume. Returns True when the turn should end."""
        self.elapsed_ms += frame_ms
        if volume > self.threshold:
            if self.first_speech_ms is None:
                self.first_speech_ms = self.elapsed_ms
            self.last_speech_ms = self.elapsed_ms
        if self.elapsed_ms >= self.max_ms:
            return True
        return (self.last_speech_ms is not None
                and self.elapsed_ms - self.last_speech_ms >= self.silence_ms)


class NoSpeechError(RuntimeError):
    pass


def record_until_silence(threshold=0.025, silence_ms=700, max_seconds=15):
    """Capture microphone frames until quiet follows detected sound.

    Leading silence is dropped except for a short pre-roll, so the first syllable isn't clipped.
    """
    detector = SilenceDetector(threshold, silence_ms, int(max_seconds * 1000))
    frames = []
    blocksize = RATE * FRAME_MS // 1000
    print("Listening... (stop speaking to finish)")
    with sd.InputStream(samplerate=RATE, channels=CHANNELS, dtype=SAMPLE_DTYPE,
                        blocksize=blocksize) as stream:
        while True:
            frame, overflowed = stream.read(blocksize)
            if overflowed:
                print("Warning: microphone input overflow")
            frames.append(frame.copy())
            if detector.update(rms(frame)):
                break
    if detector.first_speech_ms is None:
        raise NoSpeechError("No speech detected. Try a lower --threshold or speak closer to the mic.")
    first_frame = max(0, (detector.first_speech_ms - PRE_ROLL_MS) // FRAME_MS - 1)
    return np.concatenate(frames[first_frame:]).reshape(-1)


# --- STT ------------------------------------------------------------------------------

def transcribe(samples) -> str:
    """Batch STT: upload the whole utterance as WAV bytes, get text back."""
    result = openai_client.audio.transcriptions.create(
        model=VOICE_STT_MODEL,
        file=("recording.wav", to_wav_bytes(samples), "audio/wav"),
        prompt=STT_VOCABULARY,
    )
    return result.text.strip()


def listen(threshold=0.025, silence_ms=700) -> str:
    """One user turn: VAD-bounded recording, then STT."""
    samples = record_until_silence(threshold, silence_ms)
    start = time.perf_counter()
    text = transcribe(samples)
    print(f"  (STT {(time.perf_counter() - start) * 1000:.0f} ms)")
    return text


# --- TTS ------------------------------------------------------------------------------

PUNCTUATION = ".,!?;:()\"'"


def spell_id(word: str) -> str:
    """'SS201TRK' -> 'S S 2 0 1 T R K' so IDs are read character by character."""
    return " ".join(c for c in word if c.isalnum())


def is_order_or_tracking_id(word: str) -> bool:
    word = word.upper()
    return word.startswith("ORD-") or (word.startswith("SS") and word.endswith("TRK"))


def remove_list_marker(line: str) -> str:
    """'- item', '* item', '• item' or '1. item' -> 'item'."""
    if line[:2] in ("- ", "* ", "• "):
        return line[2:]
    first, _, rest = line.partition(" ")
    if first.endswith(".") and first[:-1].isdigit():
        return rest
    return line


def spell_ids(line: str) -> str:
    words = []
    for word in line.split():
        core = word.strip(PUNCTUATION)  # "SS201TRK)" -> "SS201TRK"
        if is_order_or_tracking_id(core):
            word = word.replace(core, spell_id(core))
        words.append(word)
    return " ".join(words)


def normalize_for_speech(text: str) -> str:
    """Turn chat-formatted text into something that sounds right when spoken."""
    # Markdown symbols would be read out loud.
    for symbol in ("**", "__", "`", "#"):
        text = text.replace(symbol, "")
    # "Rs 299" sounds like "R S 299"; "rupees 299" is how it's said in India.
    text = text.replace("Rs. ", "rupees ").replace("Rs ", "rupees ")

    spoken = ""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = spell_ids(remove_list_marker(line))
        if not spoken:
            spoken = line
        elif spoken.endswith((".", ",", "!", "?", ":", ";")):
            spoken += " " + line
        else:
            spoken += ". " + line  # a line break becomes a short pause
    return spoken


def _tts_args(text: str, response_format: str) -> dict:
    args = {"model": VOICE_TTS_MODEL, "voice": VOICE_TTS_VOICE,
            "input": normalize_for_speech(text), "response_format": response_format}
    if VOICE_TTS_MODEL.startswith("gpt-4o"):  # tts-1 / tts-1-hd don't accept instructions
        args["instructions"] = TTS_INSTRUCTIONS
    return args


def play_wav_bytes(data: bytes):
    with wave.open(io.BytesIO(data), "rb") as f:
        if f.getsampwidth() != 2:
            raise ValueError("Expected 16-bit PCM WAV.")
        rate, channels = f.getframerate(), f.getnchannels()
        audio = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    sd.play(audio.reshape(-1, channels), samplerate=rate)
    sd.wait()


def speak(text: str):
    """Batch TTS: download the full WAV, then play it."""
    response = openai_client.audio.speech.create(**_tts_args(text, "wav"))
    play_wav_bytes(response.content)


def speak_stream(text: str, verbose=False):
    """Streaming TTS: play PCM as it arrives, so first audio comes before the full reply.

    Ctrl+C aborts queued audio and closes the device and HTTP response.
    """
    start = time.perf_counter()
    first_audio_ms = None
    underflows = 0
    pending = b""
    with openai_client.audio.speech.with_streaming_response.create(
        **_tts_args(text, "pcm")
    ) as response:
        with sd.RawOutputStream(samplerate=TTS_PCM_RATE, channels=1, dtype="int16") as stream:
            try:
                for chunk in response.iter_bytes(chunk_size=960):  # 20 ms of PCM
                    pending += chunk
                    complete = len(pending) - len(pending) % 2  # carry a split sample over
                    if not complete:
                        continue
                    audio = np.frombuffer(pending[:complete], dtype="<i2").astype(
                        np.int16, copy=False).tobytes()
                    pending = pending[complete:]
                    if first_audio_ms is None:
                        first_audio_ms = (time.perf_counter() - start) * 1000
                    if stream.write(audio):
                        underflows += 1
                if pending:
                    raise RuntimeError("TTS returned an incomplete 16-bit PCM sample.")
                if first_audio_ms is None:
                    raise RuntimeError("No audio returned by TTS.")
            except BaseException:
                stream.abort()
                raise
    if verbose:
        print(f"First audio submitted: {first_audio_ms:.0f} ms | "
              f"request + playback: {(time.perf_counter() - start) * 1000:.0f} ms")
    if underflows:
        print(f"Warning: {underflows} speaker underflow(s); gaps may be audible.")


# --- Stage-by-stage smoke tests -------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test SnackStack's voice boundaries one at a time.")
    parser.add_argument("stage", choices=["record", "stt", "vad", "tts", "tts-stream"])
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--threshold", type=float, default=0.025)
    parser.add_argument("--silence-ms", type=int, default=700)
    parser.add_argument("--file", default="recording.wav")
    parser.add_argument("--text", default="Your order ORD-201, Butter Chicken for Rs 379, is out for delivery!")
    args = parser.parse_args()

    if args.stage == "record":
        save_wav(args.file, record_fixed(args.seconds))
        print(f"Saved {args.file}")
    elif args.stage == "vad":
        samples = record_until_silence(args.threshold, args.silence_ms)
        save_wav(args.file, samples)
        print(f"Saved {args.file} ({len(samples) / RATE:.1f} s)")
    elif args.stage == "stt":
        with wave.open(args.file, "rb") as f:
            samples = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        print(transcribe(samples))
    elif args.stage == "tts":
        print(f"Speaking: {normalize_for_speech(args.text)}")
        speak(args.text)
    elif args.stage == "tts-stream":
        print(f"Speaking: {normalize_for_speech(args.text)}")
        try:
            speak_stream(args.text, verbose=True)
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
