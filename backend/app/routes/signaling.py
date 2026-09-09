"""
Morph Backend — minimal call-signaling routes.

WS  /api/signaling/ws
    Room-based relay for WebRTC call setup between two browsers on the
    same LAN. Relays SDP offer/answer and ICE candidates; carries no
    media and performs no detection. Mirrors the JSON-over-WebSocket
    pattern used in `detection.py` (no new stack).

Contract (JSON text frames):

Client → server:
    {"type": "join",          "payload": {"room": "<code>"}}
    {"type": "offer",         "payload": {"room": "<code>", "sdp": {...}}}
    {"type": "answer",        "payload": {"room": "<code>", "sdp": {...}}}
    {"type": "ice-candidate", "payload": {"room": "<code>", "candidate": {...}}}
    {"type": "leave",         "payload": {"room": "<code>"}}

Server → client:
    {"type": "joined",      "payload": {"room": "<code>", "peers": <int>}}
    {"type": "peer-joined", "payload": {"room": "<code>", "peers": <int>}}
    {"type": "peer-left",   "payload": {"room": "<code>", "peers": <int>}}
    {"type": "offer",         "payload": {"room": "<code>", "sdp": {...}}}
    {"type": "answer",        "payload": {"room": "<code>", "sdp": {...}}}
    {"type": "ice-candidate", "payload": {"room": "<code>", "candidate": {...}}}
    {"type": "error",       "payload": {"code": <str>, "message": <str>}}

Rooms hold at most 2 peers (1:1 call demo). A third joiner gets ROOM_FULL.
SDP/candidate payloads are relayed opaquely — the server never parses them.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

log = logging.getLogger("morph-backend.signaling")

router = APIRouter(prefix="/api/signaling", tags=["signaling"])

MAX_PEERS_PER_ROOM = 2
_ROOM_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# room code (upper-cased) -> connected sockets
_rooms: dict[str, set[WebSocket]] = {}
_lock = asyncio.Lock()


def _normalize_room(raw: object) -> str | None:
    if not isinstance(raw, str):
        return None
    code = raw.strip().upper()
    if not _ROOM_RE.match(code):
        return None
    return code


async def _broadcast(room: str, message: dict, exclude: WebSocket | None = None) -> None:
    """Send message to every socket in room except `exclude`."""
    async with _lock:
        targets = [ws for ws in _rooms.get(room, set()) if ws is not exclude]
    for ws in targets:
        try:
            await ws.send_json(message)
        except Exception:
            log.warning("Signaling relay to peer failed (room=%s)", room)


async def _leave_room(room: str, ws: WebSocket) -> int:
    """Remove ws from room. Returns remaining peer count."""
    async with _lock:
        peers = _rooms.get(room)
        if peers is None:
            return 0
        peers.discard(ws)
        remaining = len(peers)
        if remaining == 0:
            _rooms.pop(room, None)
    if remaining > 0:
        await _broadcast(room, {"type": "peer-left", "payload": {"room": room, "peers": remaining}})
    return remaining


def _reset_rooms_for_tests() -> None:
    """Clear all room state. Used by tests only."""
    _rooms.clear()


@router.websocket("/ws")
async def signaling_ws(websocket: WebSocket) -> None:
    """Room-based signaling relay — one WS connection per browser."""
    await websocket.accept()
    current_room: str | None = None
    log.info("Signaling client connected")

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json(
                    {"type": "error", "payload": {"code": "BAD_MESSAGE", "message": "Message is not valid JSON"}}
                )
                continue

            msg_type = msg.get("type")
            payload = msg.get("payload") or {}

            if msg_type == "join":
                room = _normalize_room(payload.get("room"))
                if room is None:
                    await websocket.send_json(
                        {"type": "error", "payload": {"code": "BAD_ROOM", "message": "join requires payload.room (1-64 chars: A-Z 0-9 _ -)"}}
                    )
                    continue
                if current_room and current_room != room:
                    await _leave_room(current_room, websocket)
                async with _lock:
                    peers = _rooms.setdefault(room, set())
                    if websocket not in peers and len(peers) >= MAX_PEERS_PER_ROOM:
                        await websocket.send_json(
                            {"type": "error", "payload": {"code": "ROOM_FULL", "message": f"Room {room} already has {MAX_PEERS_PER_ROOM} peers"}}
                        )
                        continue
                    peers.add(websocket)
                    count = len(peers)
                current_room = room
                log.info("Signaling join room=%s peers=%d", room, count)
                await websocket.send_json({"type": "joined", "payload": {"room": room, "peers": count}})
                if count > 1:
                    await _broadcast(
                        room,
                        {"type": "peer-joined", "payload": {"room": room, "peers": count}},
                        exclude=websocket,
                    )
                continue

            if msg_type in ("offer", "answer", "ice-candidate"):
                room = _normalize_room(payload.get("room"))
                if room is None:
                    await websocket.send_json(
                        {"type": "error", "payload": {"code": "BAD_ROOM", "message": f"{msg_type} requires payload.room"}}
                    )
                    continue
                key = "sdp" if msg_type in ("offer", "answer") else "candidate"
                if payload.get(key) is None:
                    await websocket.send_json(
                        {"type": "error", "payload": {"code": "BAD_SIGNAL", "message": f"{msg_type} requires payload.{key}"}}
                    )
                    continue
                async with _lock:
                    peers = _rooms.get(room, set())
                    others = [ws for ws in peers if ws is not websocket]
                    in_room = websocket in peers
                if not in_room:
                    await websocket.send_json(
                        {"type": "error", "payload": {"code": "NOT_IN_ROOM", "message": f"Join room {room} before sending {msg_type}"}}
                    )
                    continue
                if not others:
                    # Caller is alone — tell them to wait rather than dropping silently.
                    await websocket.send_json(
                        {"type": "error", "payload": {"code": "NO_PEER", "message": "No other peer in room yet — peer must join first"}}
                    )
                    continue
                log.info("Signaling relay %s room=%s -> %d peer(s)", msg_type, room, len(others))
                await _broadcast(room, {"type": msg_type, "payload": {"room": room, key: payload[key]}}, exclude=websocket)
                continue

            if msg_type == "leave":
                room = _normalize_room(payload.get("room")) or current_room
                if room is None:
                    continue
                remaining = await _leave_room(room, websocket)
                if room == current_room:
                    current_room = None
                log.info("Signaling leave room=%s remaining=%d", room, remaining)
                continue

            await websocket.send_json(
                {"type": "error", "payload": {"code": "UNKNOWN_TYPE", "message": f"Unknown message type: {msg_type}"}}
            )

    except WebSocketDisconnect:
        log.info("Signaling client disconnected (room=%s)", current_room)
    except Exception:
        log.exception("Signaling stream error")
    finally:
        if current_room is not None:
            try:
                await _leave_room(current_room, websocket)
            except Exception:
                pass
