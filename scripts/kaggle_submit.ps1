# Kaggle CLI Unified Submission Script
# ======================================

# Configure API Token
$env:KAGGLE_API_TOKEN = "KGAT_c6ed21f79bd5ad84de5016ee4cdd2c60"

Write-Host "🚀 Starting Kaggle Submission Procedure..." -ForegroundColor Cyan

# 1. Update/Push Kernel
Write-Host "`n📁 Pushing Notebook Kernel..." -ForegroundColor Yellow
& "C:\Users\ashut\AppData\Roaming\Python\Python38\Scripts\kaggle.exe" kernels push -p notebooks/

# 2. Check for Dataset Files
if (Test-Path "checkpoint_upload\final\config.json") {
    Write-Host "`n📦 Detected model weights. Creating/Updating Kaggle Dataset..." -ForegroundColor Yellow
    # Create or update appropriately
    try {
        & "C:\Users\ashut\AppData\Roaming\Python\Python38\Scripts\kaggle.exe" datasets push -p checkpoint_upload/
    } catch {
        & "C:\Users\ashut\AppData\Roaming\Python\Python38\Scripts\kaggle.exe" datasets create -p checkpoint_upload/
    }
} else {
    Write-Host "`n⚠️  No model weights found in 'checkpoint_upload/final/'. 
Skipping dataset creation. Please add config.json, model.bin, etc., then run again." -ForegroundColor Red
}

Write-Host "`n✅ CLI Push Sequence Complete!" -ForegroundColor Green
Write-Host "Next steps:"
Write-Host "1. Go to your Writeup on Kaggle."
Write-Host "2. Attach the kernel 'tunix-gemma-reasoning-submission'."
Write-Host "3. Attach the dataset 'tunix-gemma3-1b-grpo' to your notebook."
