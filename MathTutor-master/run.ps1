## To start our streamlit app manually via powershell from the project root 

$env:PYTHONPATH = "$PSScriptRoot\src"

# ── 离线加载 HuggingFace 模型（BGE embedding 已本地缓存，避免联网超时）──
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

streamlit run "$PSScriptRoot\src\frontend\app.py"