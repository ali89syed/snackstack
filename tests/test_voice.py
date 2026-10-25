import io
import wave

import numpy as np

from snackstack.voice import SilenceDetector, normalize_for_speech, rms, save_wav, to_wav_bytes


def test_rms():
    assert rms(np.zeros(10, dtype=np.int16)) == 0
    assert rms(np.full(10, 16384, dtype=np.int16)) == 0.5


def test_silence_commits_after_speech():
    d = SilenceDetector(threshold=0.02, silence_ms=90, max_ms=1000)
    assert not d.update(0.0)
    assert not d.update(0.1)
    assert not d.update(0.0)
    assert not d.update(0.0)
    assert d.update(0.0)
    assert d.first_speech_ms == 60


def test_no_speech_until_timeout():
    d = SilenceDetector(max_ms=90)
    assert not d.update(0)
    assert not d.update(0)
    assert d.update(0)
    assert d.first_speech_ms is None


def test_wav_header(tmp_path):
    path = tmp_path / "a.wav"
    save_wav(path, np.zeros(160, dtype=np.int16))
    with wave.open(str(path)) as f:
        assert f.getframerate() == 16000
        assert f.getnchannels() == 1
        assert f.getnframes() == 160


def test_wav_bytes_round_trip():
    samples = np.arange(-50, 50, dtype=np.int16)
    with wave.open(io.BytesIO(to_wav_bytes(samples))) as f:
        assert np.array_equal(np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16), samples)


def test_normalize_for_speech():
    text = "Here you go:\n\n1. **Paneer Tikka** - Rs 199\n- Order ORD-201 (tracking SS201TRK)"
    spoken = normalize_for_speech(text)
    assert "**" not in spoken and "Rs" not in spoken
    assert spoken == "Here you go: Paneer Tikka - rupees 199. Order O R D 2 0 1 (tracking S S 2 0 1 T R K)"
