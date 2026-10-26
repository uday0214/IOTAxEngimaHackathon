"""Environment, hardware acceleration, and Google Colab integration utilities."""

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def is_google_colab() -> bool:
    """Checks whether the current Python runtime is running inside Google Colab."""
    return "google.colab" in sys.modules or os.path.exists("/content")


def is_gpu_available() -> bool:
    """Checks whether an NVIDIA GPU with CUDA acceleration is accessible."""
    # Method 1: Check PyTorch CUDA if torch is available
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return True
    except ImportError:
        pass

    # Method 2: Check nvidia-smi CLI
    if shutil.which("nvidia-smi"):
        try:
            res = subprocess.run(
                ["nvidia-smi"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
            )
            if res.returncode == 0:
                return True
        except Exception:
            pass

    return False


def get_device(requested_device: str = "auto") -> str:
    """Resolves requested device ('auto', 'cpu', 'gpu') to an active device string ('GPU' or 'CPU')."""
    req = requested_device.strip().upper()
    if req == "GPU":
        if is_gpu_available():
            return "GPU"
        logger.warning("GPU was requested, but no CUDA-enabled GPU was detected. Falling back to CPU.")
        return "CPU"
    elif req == "CPU":
        return "CPU"
    else:  # AUTO
        if is_gpu_available():
            logger.info("Auto device detection: CUDA GPU detected. Enabling GPU acceleration.")
            return "GPU"
        logger.info("Auto device detection: No GPU detected. Defaulting to CPU.")
        return "CPU"


def get_colab_secret(secret_name: str) -> Optional[str]:
    """Retrieves a secret from Google Colab userdata secrets store if running in Colab."""
    try:
        from google.colab import userdata  # type: ignore

        return userdata.get(secret_name)
    except Exception:
        return None


def get_gemini_api_key() -> Optional[str]:
    """Resolves the Gemini API key from environment variables or Google Colab Secrets."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # 1. Environment variable (e.g. .env or exported env var)
    key = os.getenv("GEMINI_API_KEY")
    if key and key.strip() and key != "your_gemini_api_key_here":
        return key.strip()

    # 2. Google Colab userdata secrets
    if is_google_colab():
        colab_key = get_colab_secret("GEMINI_API_KEY")
        if colab_key and colab_key.strip():
            logger.info("Resolved GEMINI_API_KEY securely from Google Colab userdata Secrets.")
            return colab_key.strip()

    return None


def get_system_info() -> Dict[str, Any]:
    """Collects hardware and runtime environment telemetry for diagnostics."""
    colab = is_google_colab()
    gpu = is_gpu_available()
    gpu_name = "None"

    if gpu:
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
            )
            if res.returncode == 0 and res.stdout.strip():
                gpu_name = res.stdout.strip()
        except Exception:
            gpu_name = "NVIDIA CUDA GPU (Exact model query failed)"

    return {
        "is_colab": colab,
        "is_gpu_available": gpu,
        "gpu_name": gpu_name,
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
    }


def setup_colab_drive(mount_point: str = "/content/drive", link_artifacts: bool = True) -> Optional[Path]:
    """Mounts Google Drive in Google Colab and optionally symlinks artifacts directory for persistence."""
    if not is_google_colab():
        logger.warning("setup_colab_drive was invoked outside Google Colab environment. Skipping.")
        return None

    try:
        from google.colab import drive  # type: ignore

        drive.mount(mount_point, force_remount=False)
        persistent_dir = Path(mount_point) / "MyDrive" / "AADS_Telco_Artifacts"
        persistent_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Google Drive successfully mounted at %s. Persistent folder: %s", mount_point, persistent_dir)

        if link_artifacts:
            local_artifacts = Path("artifacts")
            local_artifacts.mkdir(parents=True, exist_ok=True)
            logger.info("Artifacts persistence enabled. Target path: %s", persistent_dir)

        return persistent_dir
    except Exception as e:
        logger.warning("Failed to mount Google Drive: %s", e)
        return None
