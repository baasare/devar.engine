# AGENTS.md

## Project

Devar is an offline-first church media-team tool for live sermon
transcription, local Bible verse suggestions, and projector display control.
The repository is a monorepo with:

- `apps/frontend`: Next.js 14+ React app for the marketing page, operator
  dashboard prototype, sermon/history screens, settings, auth screens, and
  projector view.
- `apps/backend`: Django 4.2+ API for auth, service lifecycle, transcription
  orchestration, verse suggestions, projection history, configuration, Celery,
  Redis-backed SSE, and deployment support.
- `apps/playground`: Offline local prototype for transcription and verse
  suggestions. Use it to validate engine behavior without Django, Redis, Celery,
  internet access, or cloud API keys.
- `packages/devar-engine`: Shared Python engine package for audio helpers,
  transcription wrappers, transcript context extraction, Bible indexing, and
  verse suggestion.
- `packages/shared-types`: Reserved shared TypeScript type definitions.
- `packages/shared-constants`: Reserved shared constants used by frontend
  packages.
- `tools`: Scripts, Docker assets, and CI support.
- `docs`: Product, architecture, API, and deployment documentation.

## Current State

This repository is no longer skeleton-only.

- The backend has implemented Django apps, models, serializers, views, Celery
  tasks, SSE streams, migrations, tests, deployment files, and environment
  examples.
- The shared Python engine in `packages/devar-engine` is implemented and is consumed by
  both the backend and playground.
- The playground is implemented and remains the fastest place to exercise local
  transcription and verse-ranking workflows.
- The frontend exists as a Next.js/React JSX prototype with seed data and local
  UI state. Treat it as not yet fully wired to the backend API unless the code
  clearly shows otherwise.
- Some docs are still placeholders or stale. Prefer the substantive docs listed
  below and update docs when behavior changes.

## Source Of Truth

Use these documents before making architecture or implementation decisions:

- `docs/devar-dev-plan.md`
- `docs/devar-prd.md`
- `docs/devar-technical-breakdown.md`
- `docs/api.md`
- `apps/backend/docs/api.md`
- `README.md`

`docs/architecture.md` and `docs/deployment.md` are currently placeholders.
`docs/devar-ui-design.md` is referenced by older instructions but is not present
in the current repository. Do not rely on absent design docs; inspect the
existing frontend files and assets instead.

## Expected Stack

- Frontend: Next.js 14+, React 18+, currently JSX and CSS assets. TypeScript,
  TailwindCSS, SSE, and Web Audio API are planned or future integration points.
- Backend: Python 3.11+, Django 4.2+, Django REST Framework, Simple JWT,
  bcrypt password hashing, SQLite for local development, Supabase PostgreSQL for
  production, Redis, Celery, and SSE streaming responses.
- Shared engine: Python package at `packages/devar-engine`, currently using local
  transcription and local lexical/BM25-style Bible verse ranking.
- Testing: pytest/pytest-django for backend, `unittest`/pytest-compatible tests
  for the playground and shared Python package. Frontend tests are not yet
  established.
- External services: Cloud LLM and transcription providers are future optional
  enhancements. Do not introduce OpenAI, Gemini, Ollama, Whisper, or other cloud
  dependencies into the runtime path unless the user explicitly asks for that
  feature.

## Working Rules

- Keep changes scoped to the requested task.
- Preserve the monorepo boundaries between frontend, backend, shared packages, tools, and docs.
- Do not place backend logic in frontend packages or frontend-only code in backend modules.
- Prefer existing project patterns and the shared `packages/devar-engine` engine before
  creating new implementations.
- Treat the local verse suggestion path as the default baseline. Any future
  semantic provider must be optional and must preserve offline operation.
- Keep licensed Bible JSON, generated transcripts, recordings, runtime indexes,
  local SQLite databases, static build output, `.env` files, dependency
  directories, and caches out of git.
- Runtime/dependency directories may exist locally, including `node_modules/`,
  `.venv/`, `venv/`, `apps/playground/data/`,
  `apps/playground/recordings/`, and `apps/playground/transcripts/`. Ignore
  them unless the user asks for cleanup or runtime-data work.
- Do not commit secrets, API keys, local environment files, downloaded Bible
  translations, database files, build artifacts, or dependency directories.
- When adding or changing behavior that affects auth, API contracts, Celery,
  Redis/SSE flows, verse lookup, settings persistence, projection history, audio
  parsing, or service lifecycle, include focused tests or explicitly explain why
  tests were not added.
- Keep backend tests independent of microphone hardware, model downloads, Redis,
  Celery workers, internet access, and licensed Bible files unless a test is
  explicitly marked/instructed as an integration test.
- Use small fixture Bible data in tests rather than real licensed translations.

## Useful Commands

From the repository root:

```bash
python -m pip install -r requirements.txt
```

Use `requirements.txt` for local monorepo development; it keeps
`packages/devar-engine` editable. Use `requirements.prod.txt` for production or
external installs that should consume the released `devar-engine` package from
PyPI.

Backend tests:

```bash
cd apps/backend
PYTHONDONTWRITEBYTECODE=1 python -m pytest
```

Playground compatibility tests:

```bash
cd apps/playground
PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m unittest discover -s tests -v
```

Shared package tests:

```bash
cd packages/devar-engine
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

Frontend development:

```bash
cd apps/frontend
npm run dev
```

## Data And Runtime Notes

- Backend `BIBLE_DATA_DIR` defaults to `apps/playground/data`; override it with
  an absolute path in the backend environment when needed.
- Expected Bible JSON shape is a list of verse objects with `book`, `chapter`,
  `verse`, and `text`.
- Current supported translation codes are `NKJV`, `KJV`, `NIV`, and `NLT`.
- Expected audio upload format is uncompressed WAV, 16 kHz, 16-bit signed,
  mono.
- Synchronous suggestions should work without Redis/Celery. Audio upload,
  rolling transcript context, and SSE delivery depend on Redis/Celery.
