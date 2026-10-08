# AURA OS Discovery Lab

React and Tailwind frontend for generated retail dashboards, voice product search, and CLIP image search.

## Run

```powershell
cd frontend
npm install
npm run dev
```

Set `VITE_API_BASE_URL` when the API is not at `http://localhost:8000`.

The FastAPI app is `src.rag_api:app`. Configure `OPENAI_API_KEY` on the API server for GPT-4o and Whisper. Image search loads `openai/clip-vit-base-patch32` through Transformers on first use and may download model weights.
