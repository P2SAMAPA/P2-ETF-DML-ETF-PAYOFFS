"""
push_results.py  —  Upload Results to HuggingFace
"""

import os
from huggingface_hub import HfApi
import config


def upload_results(file1_path: str, file2_path: str = None, hf_token: str = None):
    """Upload results files to HuggingFace dataset."""
    token = hf_token or os.environ.get("HF_TOKEN") or config.HF_TOKEN
    if not token:
        raise ValueError("HF_TOKEN is required")

    api = HfApi(token=token)
    
    # Create repo if it doesn't exist
    try:
        api.repo_info(repo_id=config.RESULTS_REPO, repo_type="dataset")
    except:
        print(f"📦 Creating repository {config.RESULTS_REPO}...")
        api.create_repo(
            repo_id=config.RESULTS_REPO,
            repo_type="dataset",
            private=False,
            exist_ok=True
        )
    
    files_to_upload = [file1_path]
    if file2_path:
        files_to_upload.append(file2_path)
    
    for path in files_to_upload:
        if not os.path.exists(path):
            print(f"⚠️ File not found: {path}")
            continue
        filename = os.path.basename(path)
        print(f"   Uploading {filename}...")
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=filename,
            repo_id=config.RESULTS_REPO,
            token=token,
            repo_type="dataset"
        )
    print("✅ Upload complete!")


def upload_single_file(file_path: str, hf_token: str = None) -> bool:
    """Upload a single file to HuggingFace dataset."""
    try:
        upload_results(file_path, None, hf_token)
        return True
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False
