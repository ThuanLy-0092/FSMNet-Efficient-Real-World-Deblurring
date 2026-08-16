import os
from huggingface_hub import snapshot_download

os.environ["HF_HOME"] = "/data/thuanlv/hf_cache"

# Thư mục đích trên ổ cứng lớn
DEST_DIR = ""

# 2. Tải tập trung (chỉ lấy những file cần thiết)
snapshot_download(
    repo_id="rimchang/RSBlur",
    repo_type="dataset",
    local_dir=DEST_DIR,
    # Chỉ tải file RSBlur.zip và thư mục checkpoints (để test model)
    allow_patterns=["RSBlur.zip"], 
    resume_download=True,
    max_workers=16,         # 5090 và server mạnh thì để 16 cho nhanh
    local_dir_use_symlinks=False  # Copy file thật vào local_dir
)

print(f"--- Tải xong! Dữ liệu nằm tại: {DEST_DIR} ---")