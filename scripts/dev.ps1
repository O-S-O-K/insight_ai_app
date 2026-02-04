param(
    [string]$EnvName = "xai-app",
    [int]$BackendPort = 8000
)

Write-Host "Starting backend (uvicorn) in background using conda environment '$EnvName' on port $BackendPort..."
# Start uvicorn via conda in a hidden window. -NoNewWindow cannot be combined with -WindowStyle, so omit it and use -PassThru to get the process handle.
Start-Process -FilePath "conda" -ArgumentList "run -n $EnvName uvicorn api.main:app --reload --port $BackendPort" -WindowStyle Hidden -PassThru | Out-Null

Start-Sleep -Seconds 2

$backendUrl = "http://localhost:$BackendPort"
Write-Host "Setting INSIGHT_BACKEND_URL environment variable for this session..."
$env:INSIGHT_BACKEND_URL = $backendUrl

Write-Host "Starting Streamlit frontend in foreground (press Ctrl+C to stop)..."
conda run -n $EnvName streamlit run app/app.py
