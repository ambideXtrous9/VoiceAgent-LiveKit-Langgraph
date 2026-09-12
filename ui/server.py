#!/usr/bin/env python3
"""
Lightweight Web Server for the LiveKit Voice Agent UI.
Serves the aesthetic web frontend and provides LiveKit WebRTC access tokens.
"""

import logging
import os
import sys
import uuid
from pathlib import Path
from aiohttp import web
from dotenv import find_dotenv, load_dotenv
from livekit import api

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment configuration
load_dotenv(find_dotenv())

logger = logging.getLogger("livekit.ui_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
PORT = int(os.getenv("PORT", 7860))

STATIC_DIR = os.path.dirname(os.path.abspath(__file__))


async def handle_token(request: web.Request) -> web.Response:
    """Generate and return a LiveKit access token for joining a room."""
    if not LIVEKIT_URL or not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
        logger.error("LiveKit credentials missing in environment.")
        return web.json_response(
            {"error": "LiveKit credentials not configured in .env"},
            status=500,
        )

    try:
        # Sanitize query parameters or use defaults
        raw_room = request.query.get("room", "").strip()
        room_name = raw_room if raw_room else f"voice-{uuid.uuid4().hex[:6]}"
        
        raw_name = request.query.get("name", "").strip()
        participant_name = raw_name if raw_name else "User"
        
        identity = f"caller-{uuid.uuid4().hex[:4]}"

        token = (
            api.AccessToken(api_key=LIVEKIT_API_KEY, api_secret=LIVEKIT_API_SECRET)
            .with_identity(identity)
            .with_name(participant_name)
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=room_name,
                    can_publish=True,
                    can_subscribe=True,
                    can_publish_data=True,
                )
            )
        )

        jwt_token = token.to_jwt()
        logger.info("Issued token for identity '%s' (name='%s') in room '%s'", identity, participant_name, room_name)

        return web.json_response({
            "token": jwt_token,
            "url": LIVEKIT_URL,
            "room": room_name,
            "identity": identity,
            "name": participant_name,
        })
    except Exception as e:
        logger.exception("Failed to issue LiveKit access token: %s", e)
        return web.json_response(
            {"error": f"Failed to generate access token: {str(e)}"},
            status=500,
        )


async def handle_health(request: web.Request) -> web.Response:
    """Health check endpoint for container environments and monitoring."""
    return web.json_response({
        "status": "healthy",
        "service": "livekit-voice-agent-ui",
        "livekit_configured": bool(LIVEKIT_URL and LIVEKIT_API_KEY and LIVEKIT_API_SECRET),
    })


async def handle_index(request: web.Request) -> web.FileResponse:
    """Serve the single-page voice visualizer application."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        return web.Response(text="index.html not found", status=404)
    return web.FileResponse(index_path)


def create_app() -> web.Application:
    """Factory creating the aiohttp application."""
    app = web.Application()
    app.router.add_get("/", handle_index)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/api/token", handle_token)
    app.router.add_static("/static", STATIC_DIR)
    return app


if __name__ == "__main__":
    app = create_app()
    logger.info("=================================================================")
    logger.info("🎙️  Starting Voice Agent UI Server on http://localhost:%d", PORT)
    logger.info("   • Health check: http://localhost:%d/health", PORT)
    logger.info("   • Token endpoint: http://localhost:%d/api/token", PORT)
    logger.info("=================================================================")
    web.run_app(app, host="0.0.0.0", port=PORT)
