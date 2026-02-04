Local development

1) Create a Python environment (recommended: venv)

2) Install dependencies:

   pip install -r api/requirements.txt

3) Set environment variables (optional for BLIP captions):

   export HF_TOKEN=your_hf_token_here
   export INSIGHT_BACKEND_URL=http://localhost:8000

4) Run the backend:

   uvicorn api.main:app --reload --port 8000

5) Run the Streamlit app (in another terminal):

   export INSIGHT_BACKEND_URL=http://localhost:8000
   streamlit run app/app.py

This configuration will run the heavy models (TensorFlow) in the backend while the Streamlit frontend calls it for predictions, grad-cam and captions.

---

## Deployment (Docker/Render)

For cloud deployment, this backend is designed to run on [Render](https://render.com/) or any Docker-compatible host. See the included `Dockerfile` and `DEPLOY_RENDER.md` for details.

**Environment variables:**
- `HF_TOKEN` (optional, for BLIP captions)
- `INSIGHT_BACKEND_URL` (set by frontend to point to backend)

For overall project context and frontend setup, see the [root README](../README.md).