#!/usr/bin/env python3
"""
Lightweight Web Server for the LiveKit Voice Agent UI.
Serves the aesthetic web frontend and provides LiveKit WebRTC access tokens.
"""

import os
import uuid
import logging
from aiohttp import web
from dotenv import find_dotenv, load_dotenv
from livekit import api

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
    """Health check endpoint providing service status and configuration diagnostics."""
    return web.json_response({
        "status": "healthy",
        "livekit_configured": bool(LIVEKIT_URL and LIVEKIT_API_KEY and LIVEKIT_API_SECRET),
        "livekit_url": LIVEKIT_URL or "not_configured",
        "port": PORT,
    })


async def handle_index(request: web.Request) -> web.FileResponse:
    """Serve the single-page application index.html."""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_file):
        raise web.HTTPNotFound(text="UI index.html not found.")
    return web.FileResponse(index_file)


def create_app() -> web.Application:
    """Construct and configure the aiohttp web application with route handlers."""
    app = web.Application()
    app.router.add_get("/", handle_index)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/api/token", handle_token)
    app.router.add_static("/static/", path=STATIC_DIR, name="static")
    return app


if __name__ == "__main__":
    app = create_app()
    print("=" * 60)
    print(f"🎙️  LiveKit Voice Agent Web UI Server")
    print(f"🔗  Open in Browser: http://localhost:{PORT}")
    print(f"🩺  Health Check  : http://localhost:{PORT}/health")
    print("=" * 60)
    web.run_app(app, host="0.0.0.0", port=PORT, print=None)
