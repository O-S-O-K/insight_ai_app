# Quick Start - Render Deployment

## When Your Render Free Instances Reset

### 1. Configure Environment Variable
1. Go to https://dashboard.render.com
2. Click on `insight-ai-app` service
3. Go to **Environment** tab
4. Click **Add Environment Variable**
5. Add:
   - **Key**: `ENABLE_BLIP`
   - **Value**: `false`
6. Click **Save Changes**

### 2. Wait for Deployment
- Render will automatically redeploy (~5-10 minutes)
- Monitor the deployment logs if needed

### 3. Verify Deployment
Check the health endpoint:
```bash
curl https://insight-ai-app.onrender.com/
```

Should return:
```json
{
  "status": "ready",
  "blip_enabled": false,
  "model_loaded": true
}
```

### 4. Test Your App
Go to: https://insight-ai-v1.streamlit.app

Test features:
- ✅ **Predictions**: Should show specific breeds (e.g., "German Shepherd", not just "dog")
- ✅ **Grad-CAM**: Should highlight correct regions
- ❌ **Captions**: Will show disabled message (expected with free tier)

---

## What's Working

| Feature | Status |
|---------|--------|
| Specific breed predictions | ✅ Working |
| Accurate Grad-CAM heatmaps | ✅ Working |
| Human feedback | ✅ Working |
| BLIP captions | ❌ Disabled (to fit in 512 MB) |

---

## If You Want Captions Too

Upgrade to Render Starter plan ($7/month):
1. Upgrade your service to 2 GB RAM
2. Remove `ENABLE_BLIP` variable or set it to `true`
3. Redeploy

---

## Need More Details?

See `DEPLOYMENT_SUMMARY.md` for complete technical documentation.
