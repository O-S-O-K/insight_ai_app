Deploying the InsightAI backend to Render (free tier)

1) Create a new Web Service on Render
   - Connect your GitHub repo and pick the `api/` folder (use the Dockerfile-based service) or set "Build Command" to `pip install -r api/requirements.txt` and "Start Command" to `uvicorn api.main:app --host 0.0.0.0 --port 8000`.

2) Set environment variables in Render:
   - `HF_TOKEN` — (required for BLIP captioning via Hugging Face Inference API)
   - Optional: `CORS_ALLOWED_ORIGINS` or other secret keys

3) Build & Deploy
   - Wait for the build to finish. Render will build/install TensorFlow and the rest (this can take a few minutes).

4) After deployment
   - Note the service URL (e.g., `https://insight-backend.onrender.com`).
   - In your Streamlit deployment (or local `app`), set `INSIGHT_BACKEND_URL` to that URL.

Notes
- Render free tier provides ephemeral filesystem storage — fine for demo feedback and images, but not durable for long-term storage. For production, use S3 or a persistent DB.
- If BLIP captioning is desired without running `transformers`/`torch` on your own backend, ensure `HF_TOKEN` is set so the backend can call Hugging Face Inference endpoint.
