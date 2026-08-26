# Single-container deployment: builds the React frontend, then serves it
# and the FastAPI backend from one process on one port -- the shape a
# platform that exposes exactly one port per app (e.g. Hugging Face
# Spaces' Docker SDK) needs. Local development is unaffected by this file
# -- it still runs `npm run dev` (Vite, :5173) and
# `uvicorn election_api:app --reload` (:8000) as two separate processes;
# see README's "Interactive app" section.

FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim AS runtime
WORKDIR /app

# Real, checked-in data only (attrition/data/*.json) -- no raw manual
# downloads (repo-root data/, gitignored and dockerignored) are needed at
# runtime; every module that would read them only runs at `scripts/
# process_*.py` time, not from the API.
COPY pyproject.toml README.md ./
COPY attrition/ ./attrition/
COPY experiments/ ./experiments/
COPY election_api.py ./
RUN pip install --no-cache-dir -e ".[api,llm,groq]"

COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Hugging Face Spaces' Docker SDK expects the app to listen on $PORT
# (defaults to 7860 if unset, matching the Space's own default).
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn election_api:app --host 0.0.0.0 --port ${PORT:-7860}"]
