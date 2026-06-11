"""
Push hindi_slm_v001 to HuggingFace Hub.

Usage:
    python push_to_hub.py                         # uses cached hf token
    python push_to_hub.py --token hf_xxx...       # explicit token
    python push_to_hub.py --repo myname/my-repo   # custom repo name
"""

import argparse
import sys
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "artifacts" / "models" / "hindi_slm_v001"
DEFAULT_REPO = "vaibhavmaurya/hindi-slm-v001"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_REPO, help="HF repo id (user/name)")
    parser.add_argument("--token", default=None, help="HF token (optional if already logged in)")
    parser.add_argument("--private", action="store_true", help="Create as private repo")
    args = parser.parse_args()

    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=args.token)

    # Verify login
    try:
        me = api.whoami()
        print(f"Logged in as: {me['name']}")
    except Exception as e:
        print(f"ERROR: Not logged in. Run `huggingface-cli login` or pass --token.")
        print(f"Details: {e}")
        sys.exit(1)

    # Create repo if it doesn't exist
    print(f"\nCreating repo '{args.repo}' (private={args.private}) ...")
    create_repo(
        repo_id=args.repo,
        repo_type="model",
        private=args.private,
        exist_ok=True,
        token=args.token,
    )
    print(f"  Repo ready: https://huggingface.co/{args.repo}")

    # Upload entire model directory
    print(f"\nUploading from {MODEL_DIR} ...")
    api.upload_folder(
        folder_path=str(MODEL_DIR),
        repo_id=args.repo,
        repo_type="model",
        commit_message="Upload hindi-slm-v001: 46M param Hindi causal LM, step 50k, PPL=38.63",
    )

    print(f"\nDone!  https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
