"""Regression tests for the minimal call-signaling WebSocket.

Spins up the real uvicorn app in a background thread and drives
/api/signaling/ws like the frontend does: two clients join the same
room, one sends offer + ICE, the other receives them, answers, and the
first receives the answer. No model or audio data needed.
"""

import asyncio
import json
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest
import uvicorn

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.main import app  # noqa: E402
from app.routes import signaling as signaling_route  # noqa: E402


class _Server:
    def __init__(self) -> None:
        self.config = uvicorn.Config(app, host="127.0.0.1", port=8012, log_level="error")
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


@pytest.fixture(autouse=True)
def _clean_rooms():
    signaling_route._reset_rooms_for_tests()
    yield
    signaling_route._reset_rooms_for_tests()


def _room() -> str:
    return "T" + uuid.uuid4().hex[:7].upper()


def test_offer_answer_ice_relay(server) -> None:
    import websockets

    room = _room()
    offer = {"type": "offer", "sdp": "fake-offer-sdp"}
    answer = {"type": "answer", "sdp": "fake-answer-sdp"}
    candidate = {"type": "ice", "candidate": "fake-candidate"}

    async def run() -> None:
        async with (
            websockets.connect("ws://127.0.0.1:8012/api/signaling/ws") as a,
            websockets.connect("ws://127.0.0.1:8012/api/signaling/ws") as b,
        ):
            await a.send(json.dumps({"type": "join", "payload": {"room": room}}))
            msg = json.loads(await asyncio.wait_for(a.recv(), 10))
            assert msg["type"] == "joined" and msg["payload"]["peers"] == 1

            await b.send(json.dumps({"type": "join", "payload": {"room": room}}))
            msg = json.loads(await asyncio.wait_for(b.recv(), 10))
            assert msg["type"] == "joined" and msg["payload"]["peers"] == 2
            # First peer is notified the second joined.
            msg = json.loads(await asyncio.wait_for(a.recv(), 10))
            assert msg["type"] == "peer-joined" and msg["payload"]["peers"] == 2

            # A offers, B receives.
            await a.send(json.dumps({"type": "offer", "payload": {"room": room, "sdp": offer}}))
            msg = json.loads(await asyncio.wait_for(b.recv(), 10))
            assert msg["type"] == "offer" and msg["payload"]["sdp"] == offer

            # B answers, A receives.
            await b.send(json.dumps({"type": "answer", "payload": {"room": room, "sdp": answer}}))
            msg = json.loads(await asyncio.wait_for(a.recv(), 10))
            assert msg["type"] == "answer" and msg["payload"]["sdp"] == answer

            # ICE both directions.
            await a.send(json.dumps({"type": "ice-candidate", "payload": {"room": room, "candidate": candidate}}))
            msg = json.loads(await asyncio.wait_for(b.recv(), 10))
            assert msg["type"] == "ice-candidate" and msg["payload"]["candidate"] == candidate

            await b.send(json.dumps({"type": "ice-candidate", "payload": {"room": room, "candidate": candidate}}))
            msg = json.loads(await asyncio.wait_for(a.recv(), 10))
            assert msg["type"] == "ice-candidate"

    asyncio.run(run())


def test_room_full_and_bad_room(server) -> None:
    import websockets

    room = _room()

    async def run() -> None:
        async with (
            websockets.connect("ws://127.0.0.1:8012/api/signaling/ws") as a,
            websockets.connect("ws://127.0.0.1:8012/api/signaling/ws") as b,
            websockets.connect("ws://127.0.0.1:8012/api/signaling/ws") as c,
        ):
            for ws in (a, b):
                await ws.send(json.dumps({"type": "join", "payload": {"room": room}}))
                await asyncio.wait_for(ws.recv(), 10)  # joined
            await asyncio.wait_for(a.recv(), 10)  # peer-joined
            await c.send(json.dumps({"type": "join", "payload": {"room": room}}))
            msg = json.loads(await asyncio.wait_for(c.recv(), 10))
            assert msg["type"] == "error" and msg["payload"]["code"] == "ROOM_FULL"

            await c.send(json.dumps({"type": "join", "payload": {"room": ""}}))
            msg = json.loads(await asyncio.wait_for(c.recv(), 10))
            assert msg["type"] == "error" and msg["payload"]["code"] == "BAD_ROOM"

    asyncio.run(run())


def test_offer_without_peer_reports_no_peer(server) -> None:
    import websockets

    room = _room()

    async def run() -> None:
        async with websockets.connect("ws://127.0.0.1:8012/api/signaling/ws") as a:
            await a.send(json.dumps({"type": "join", "payload": {"room": room}}))
            await asyncio.wait_for(a.recv(), 10)  # joined
            await a.send(json.dumps({"type": "offer", "payload": {"room": room, "sdp": {"x": 1}}}))
            msg = json.loads(await asyncio.wait_for(a.recv(), 10))
            assert msg["type"] == "error" and msg["payload"]["code"] == "NO_PEER"

    asyncio.run(run())
