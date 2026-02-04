Local development helper

Use the PowerShell helper to run the backend and frontend together for quick testing.

Start both (backend in background + frontend in foreground):

```powershell
# From project root
./scripts/dev.ps1 -EnvName xai-app -BackendPort 8000
```

This:
- starts `uvicorn api.main:app` in the background (via `conda run -n xai-app`)
- sets `INSIGHT_BACKEND_URL` for that session
- runs `streamlit run app/app.py` in the foreground (Ctrl+C to stop)

Stop background services if needed:

```powershell
./scripts/dev-stop.ps1
```

Notes:
- The scripts use `conda run -n <env>` so you don't need to activate the environment before running, but ensure the environment exists and dependencies are installed.
- If `conda` is not on PATH, open a conda-enabled PowerShell (Anaconda Prompt) before running the scripts.
- For macOS/Linux you can use equivalent shell helpers (not included).
