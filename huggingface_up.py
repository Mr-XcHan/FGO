from huggingface_hub import create_repo, upload_folder

import os
token = os.environ["HF_TOKEN"]


username = "HHHHxc"
repo_name = "DeepSeek-R1-Distill-Qwen-1.5B-CEWE"
repo_id = f"{username}/{repo_name}"

# 创建私有模型仓库
create_repo(repo_id, repo_type="model", private=True, token=token)

# 上传模型文件夹
upload_folder(
    folder_path="/mnt/fast/nobackup/scratch4weeks/ly0008/xch/code/CEWE_/output_logs/CEWE/GRPO/DeepSeek-R1-Distill-Qwen-1.5B/checkpoint-200",
    repo_id=repo_id,
    repo_type="model",
    token=token
)
