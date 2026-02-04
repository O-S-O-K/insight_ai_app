Write-Host "Stopping uvicorn (if running)..."
Get-Process -Name uvicorn -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }

Write-Host "Stopping Streamlit (if running)..."
Get-Process -Name streamlit -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }

Write-Host "Done."