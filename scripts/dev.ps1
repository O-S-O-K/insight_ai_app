param(
    [string]$EnvName = "xai-app",
    [int]$BackendPort = 8000
)

Write-Host "Starting backend (uvicorn) in background using conda environment '$EnvName' on port $BackendPort..."
Start-Process -NoNewWindow -FilePath "conda" -ArgumentList "run -n $EnvName uvicorn api.main:app --reload --port $BackendPort" -WindowStyle Hidden

Start-Sleep -Seconds 2

$backendUrl = "http://localhost:$BackendPort"
Write-Host "Setting INSIGHT_BACKEND_URL environment variable for this session..."
$env:INSIGHT_BACKEND_URL = $backendUrl

Write-Host "Starting Streamlit frontend in foreground (press Ctrl+C to stop)..."
conda run -n $EnvName streamlit run app/app.py
