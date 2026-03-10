from huggingface_hub import create_repo, upload_folder

import os
token = os.environ["HF_TOKEN"]


username = ""
repo_name = ""
repo_id = f"{username}/{repo_name}"

# 创建私有模型仓库
create_repo(repo_id, repo_type="model", private=True, token=token)

# 上传模型文件夹
upload_folder(
    folder_path="",
    repo_id=repo_id,
    repo_type="model",
    token=token
)
