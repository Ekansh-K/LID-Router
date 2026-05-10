"""
Helper script to download and cache models locally before running the UI.
This ensures the UI backend starts up quickly.
"""
import sys
import torch
from src.pipeline import Pipeline
from src.utils import get_logger

log = get_logger("download_models")

if __name__ == "__main__":
    print("=" * 70)
    print("LID Router - GPU Utilization Check")
    print("=" * 70)
    if torch.cuda.is_available():
        print("[OK] CUDA GPU detected. Models will be loaded onto the GPU.")
    else:
        print("[WARNING] CUDA GPU NOT DETECTED. PyTorch is running in CPU-only mode.")
        print("To utilize your local GPU, you must install the CUDA version of PyTorch.")
        print("Run the following command in your terminal:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("=" * 70)

    log.info("Initializing pipeline to download models...")
    log.info("This will download ~10GB of models to ~/.cache/huggingface if not present.")
    
    pipe = Pipeline(routing_policy="learned")
    pipe.load_models(sequential=False)
    
    log.info("All models successfully downloaded and loaded into memory!")
    log.info("You can now run: uvicorn ui.app:app --reload")
