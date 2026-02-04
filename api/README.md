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