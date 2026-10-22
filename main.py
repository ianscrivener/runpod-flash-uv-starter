
"""
runpod flash seed
"""

import asyncio
from pathlib import Path

import yaml
from dotenv import load_dotenv
from runpod_flash import Endpoint, GpuGroup


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# populate os.environ from .env
load_dotenv(Path(__file__).parent / ".env")  

# load config.yaml once at import time; drives GPU selection + worker deps below
_cfg_path = Path(__file__).parent / "config.yaml"
with open(_cfg_path) as f:
    CFG = yaml.safe_load(f)

# resolve GPU group string -> enum member
_GPU  = getattr(GpuGroup, CFG["flash"]["gpu"])  


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

# runs remotely on the Flash worker; the imports below are inside the function
# because this code executes in the worker's environment, not locally
@Endpoint(
    name="gpu-info",
    gpu=_GPU,
    workers=1,
    execution_timeout_ms=0,
)
async def hello():
    import subprocess

    # sanity-check the worker: confirm GPU visibility and mounted volume size
    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    du = subprocess.run(["du", "-sh", "/workspace"], capture_output=True, text=True).stdout

    return {"nvidia_smi": smi, "workspace_disk_usage": du}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    # local driver: submits the @Endpoint job to RunPod Flash and prints the result
    hello_result = await hello()
    print(hello_result["nvidia_smi"])
    print(hello_result["workspace_disk_usage"])


if __name__ == "__main__":
    asyncio.run(main())
