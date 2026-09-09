"""Regression tests for the live-streaming detection WebSocket.

Spins up the real uvicorn app in a background thread (in-process) and
drives /api/detection/ws like the frontend does: streaming mono float32
PCM chunks at 16 kHz (the model's native rate), expecting detection_result
payloads per completed window.

Skips when the gary_stafford data files are absent.
"""

import asyncio
import base64
import json
import sys
import threading
import time
from pathlib import Path

import librosa
import numpy as np
import pytest
import uvicorn

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.main import app  # noqa: E402

SR = 16000
DATA = Path(r"C:\Users\Akasdip\morph\data\raw\gary_stafford")
REAL_WAV = DATA / "real" / "gs_00202.wav"
FAKE_WAV = DATA / "fake" / "gs_01101.wav"

_needs_data = pytest.mark.skipif(
    not REAL_WAV.exists() or not FAKE_WAV.exists(),
    reason="gary_stafford samples unavailable",
)


class _Server:
    def __init__(self) -> None:
        self.config = uvicorn.Config(app, host="127.0.0.1", port=8011, log_level="error")
        self.server = uvicorn.Server(self.config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self) -> "_Server":
        self.thread.start()
        deadline = time.time() + 30
        while not self.server.started and time.time() < deadline:
            time.sleep(0.05)
        if not self.server.started:
            raise RuntimeError("server failed to start")
        return self

    def __exit__(self, *exc) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


@pytest.fixture(scope="module")
def server():
    with _Server() as s:
        yield s


def pcm_b64(samples: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(samples, dtype="<f4").tobytes()).decode("ascii")


def _stream_and_collect(path: Path, want: str) -> list[dict]:
    import websockets

    y, _ = librosa.load(str(path), sr=SR, mono=True)
    n = SR // 2

    async def run() -> list[dict]:
        results = []
        async with websockets.connect("ws://127.0.0.1:8011/api/detection/ws") as ws:
            first = json.loads(await asyncio.wait_for(ws.recv(), 10))
            assert first["type"] == "status_update"
            for i in range(0, len(y), n):
                await ws.send(
                    json.dumps(
                        {
                            "type": "audio_chunk",
                            "payload": {
                                "data": pcm_b64(y[i : i + n]),
                                "sample_rate": SR,
                                "timestamp": int(time.time() * 1000),
                            },
                        }
                    )
                )
            deadline = time.time() + 60
            while time.time() < deadline:
                msg = json.loads(await asyncio.wait_for(ws.recv(), 30))
                if msg["type"] == "detection_result":
                    results.append(msg["payload"])
                    if want in {p["label_str"] for p in results}:
                        break
                elif msg["type"] == "error":
                    raise AssertionError(f"server error: {msg['payload']}")
        return results

    return asyncio.run(run())


@_needs_data
def test_ws_real_stream(server) -> None:
    results = _stream_and_collect(REAL_WAV, "REAL")
    assert results, "no detection_result received"
    assert any(r["label_str"] == "REAL" for r in results)
    for r in results:
        assert {"label", "label_str", "confidence", "real_probability",
                "fake_probability", "chunk_duration"} <= r.keys()
        assert 0.0 <= r["confidence"] <= 1.0
        assert r["chunk_duration"] > 0


@_needs_data
def test_ws_fake_stream(server) -> None:
    results = _stream_and_collect(FAKE_WAV, "FAKE")
    assert results, "no detection_result received"
    assert any(r["label_str"] == "FAKE" for r in results)


def test_ws_bad_sample_rate(server) -> None:
    import websockets

    async def run() -> None:
        async with websockets.connect("ws://127.0.0.1:8011/api/detection/ws") as ws:
            await asyncio.wait_for(ws.recv(), 10)  # connected
            await asyncio.wait_for(ws.recv(), 10)  # idle
            await ws.send(
                json.dumps(
                    {
                        "type": "audio_chunk",
                        "payload": {"data": pcm_b64(np.zeros(32000, dtype="f4")), "sample_rate": 12345},
                    }
                )
            )
            msg = json.loads(await asyncio.wait_for(ws.recv(), 10))
            assert msg["type"] == "error"
            assert msg["payload"]["code"] == "BAD_SAMPLE_RATE"

    asyncio.run(run())


def test_ws_bad_audio_length(server) -> None:
    import websockets

    async def run() -> None:
        async with websockets.connect("ws://127.0.0.1:8011/api/detection/ws") as ws:
            await asyncio.wait_for(ws.recv(), 10)
            await asyncio.wait_for(ws.recv(), 10)
            await ws.send(
                json.dumps(
                    {
                        "type": "audio_chunk",
                        "payload": {
                            "data": base64.b64encode(b"\x00\x01").decode("ascii"),  # 2 bytes, not 4-aligned
                            "sample_rate": SR,
                        },
                    }
                )
            )
            msg = json.loads(await asyncio.wait_for(ws.recv(), 10))
            assert msg["type"] == "error"
            assert msg["payload"]["code"] == "BAD_AUDIO"

    asyncio.run(run())


def test_decode_pcm_base64_unit() -> None:
    from app.routes.detection import _decode_pcm_base64

    y = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype="<f4")
    out = _decode_pcm_base64(base64.b64encode(y.tobytes()).decode("ascii"))
    np.testing.assert_allclose(out, y, rtol=1e-6)

    with pytest.raises(ValueError, match="4"):
        _decode_pcm_base64(base64.b64encode(b"\x00\x01").decode("ascii"))
    with pytest.raises(ValueError, match="base64"):
        _decode_pcm_base64("not base64!!!")