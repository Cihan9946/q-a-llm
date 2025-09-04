from huggingface_hub import snapshot_download
import os

# Hugging Face erişim token'ını ayarla
os.environ["HUGGINGFACE_HUB_TOKEN"] = "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"

# İndirilecek model ID'si ve hedef klasör
model_id = "google/gemma-2b-it"
local_dir = "./local_models/gemma-2b-it"

# Modeli indir (cache değil, gerçek dosyalarla)
snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    token=os.environ["HUGGINGFACE_HUB_TOKEN"]
)
