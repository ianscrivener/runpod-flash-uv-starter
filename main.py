
"""
runpod flash seed
"""

from pathlib import Path

import runpod
import yaml
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

if runpod.is_local():
    # populate os.environ from .env
    load_dotenv(Path(__file__).parent / ".env")

    # load config.yaml once at import time. It drives GPU selection + worker deps below
    _cfg_path = Path(__file__).parent / "config.yaml"
    with open(_cfg_path) as f:
        CFG = yaml.safe_load(f)
else:
    # the worker runs this module again without .env or config.yaml.
    # Only the local machine uses the decorator arguments
    CFG = {"flash": {"gpu": "ANY", "workers": 1, "min_cuda_version": None}}

# resolve GPU group string -> enum member
_GPU = getattr(runpod.GpuGroup, CFG["flash"]["gpu"])

# the app groups the resources below when you deploy
app = runpod.App("mflux-model-maker")


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

# runs on the remote worker. The imports below are inside the function
# because this code executes in the worker's environment, not locally
@app.queue(
    name="gpu-info",
    gpu=_GPU,
    workers=(0, CFG["flash"]["workers"]),
    execution_timeout_ms=0,
    min_cuda_version=CFG["flash"]["min_cuda_version"],
)
def hello():
    import subprocess

    # sanity-check the worker: confirm GPU visibility and mounted volume size
    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    du = subprocess.run(["du", "-sh", "/workspace"], capture_output=True, text=True).stdout

    return {"nvidia_smi": smi, "workspace_disk_usage": du}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# run with `rp flash dev main.py --once`
@runpod.local_entrypoint
def main():
    # local driver: submits the job to the worker and prints the result
    hello_result = hello.remote()
    print(hello_result["nvidia_smi"])
    print(hello_result["workspace_disk_usage"])
