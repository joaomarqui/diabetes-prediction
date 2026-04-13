# ============================================================
# Dockerfile - Diabetes Prediction API
# ============================================================
# Two-phase copy pattern for optimal Docker layer caching:
#   Phase 1: copy lock files -> install dependencies (cached)
#   Phase 2: copy source code -> install project (fast)
#
# conf/ and data/ are NOT baked into the image.
# They are mounted at runtime via docker-compose volumes.
# ============================================================

FROM python:3.13-slim

# Copy uv binary from official image (10-100x faster than pip)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Pre-compile .py to .pyc for faster cold-start
ENV UV_COMPILE_BYTECODE=1

WORKDIR /app

# --- Phase 1: Install dependencies (cached layer) ---
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev --no-install-project 2>/dev/null || \
    uv sync --no-dev --no-install-project

# --- Phase 2: Install project code ---
COPY src/ src/
RUN uv sync --no-dev 2>/dev/null || uv sync --no-dev

# Document port (actual publishing done in docker-compose)
EXPOSE 8000

# Start FastAPI with uvicorn
CMD ["uv", "run", "uvicorn", "diabetes.api:app", "--host", "0.0.0.0", "--port", "8000"]
