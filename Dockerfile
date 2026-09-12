# syntax=docker/dockerfile:1

# ==============================================================================
# Stage 1: Build & Dependency Resolution
# ==============================================================================
FROM python:3.13-slim AS builder

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Configure uv to install into dedicated virtual environment
ENV UV_PROJECT_ENVIRONMENT="/opt/venv" \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Copy dependency manifests for layer caching
COPY pyproject.toml uv.lock ./

# Install production dependencies
RUN uv sync --frozen --no-install-project --no-dev


# ==============================================================================
# Stage 2: Production Runtime
# ==============================================================================
FROM python:3.13-slim AS runner

WORKDIR /app

# Install runtime utilities (curl for container healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built virtual environment
COPY --from=builder /opt/venv /opt/venv

# Configure runtime environment
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app" \
    PORT=7860

# Create non-root application user
RUN useradd -m -u 1001 appuser && \
    mkdir -p /home/appuser/.cache && \
    chown -R appuser:appuser /app /home/appuser

# Copy application source code
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser web/ web/
COPY --chown=appuser:appuser pyproject.toml .

# Switch to non-root user
USER appuser

# Pre-download Silero VAD and TurnDetector model weights during build
RUN python src/agent.py download-files

# Expose Web UI & Token Server port
EXPOSE 7860

# Default entry: start web server (overridable in docker-compose)
CMD ["python", "web/server.py"]
