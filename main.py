"""
runpod flash seed
"""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import yaml
from dotenv import load_dotenv
from runpod_flash import Endpoint, GpuGroup

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).parent / ".env")

_cfg_path = Path(__file__).parent / "config.yaml"
with open(_cfg_path) as f:
    CFG = yaml.safe_load(f)

_GPU  = getattr(GpuGroup, CFG["flash"]["gpu"])
_DEPS = CFG["python"]["dependencies"] + ["huggingface_hub", "jinja2", "pyyaml", "python-dotenv"]

# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

@Endpoint(
    name="convert-perplexity",
    gpu=_GPU,
    workers=1,
    # dependencies=_DEPS,
    dependencies=["subprocess","huggingface_hub","jinja2","pyyaml","python-dotenv","mlx[cuda]","mlx_lm","datasets"],  # uv_dependencies doesn't work for some reason, so hardcoding here
    # python_version=CFG["python"]["version"],
    env={"HF_TOKEN": os.environ["HF_TOKEN"]},
    execution_timeout_ms=0,
)
async def run_pipeline(cfg: dict):
    import os
    import sys
    import subprocess
    from concurrent.futures import ThreadPoolExecutor
    from datetime import date
    from pathlib import Path
    from huggingface_hub import snapshot_download, HfApi
    from jinja2 import Template

    hf_token   = os.environ["HF_TOKEN"]
    model_id   = cfg["model_id"]
    output_repo = cfg["output_repo"]
    q          = cfg["quant"]
    p          = cfg["perplexity"]

    work_dir    = Path("/tmp/convert")
    model_dir   = work_dir / "source_model"
    quant_dir   = work_dir / "quantized_model"
    dataset_dir = work_dir / "dataset"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Step 1 — parallel downloads
    print("Step 1 — downloading model and dataset in parallel...")
    def dl_model():
        snapshot_download(
            repo_id=model_id,
            local_dir=str(model_dir),
            token=hf_token,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
        )

    def dl_dataset():
        snapshot_download(
            repo_id=p["data_path"],
            repo_type="dataset",
            local_dir=str(dataset_dir),
            token=hf_token,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_model   = pool.submit(dl_model)
        fut_dataset = pool.submit(dl_dataset)
        fut_model.result()
        fut_dataset.result()
    print("  downloads complete")

    # Step 2 — AWQ quantize
    print("Step 2 — quantizing...")
    quant_cmd = [
        sys.executable, "-m", "mlx_lm", "awq",
        "--model",            str(model_dir),
        "--mlx-path",         str(quant_dir),
        "--bits",             str(q["bits"]),
        "--group-size",       str(q["group_size"]),
        "--embed-bits",       str(q["embed_bits"]),
        "--embed-group-size", str(q["embed_group_size"]),
        "--num-samples",      str(q["num_samples"]),
        "--sequence-length",  str(q["sequence_length"]),
        "--n-grid",           str(q["n_grid"]),
    ]
    result = subprocess.run(quant_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Quantization failed:\n{result.stderr}")
    print("  quantization complete")

    # Step 3a — perplexity
    print("Step 3 — running perplexity test...")
    ppl_cmd = [
        sys.executable, "-m", "mlx_lm", "perplexity",
        "--model",           str(quant_dir),
        "--data-path",       str(dataset_dir),
        "--batch-size",      str(p["batch_size"]),
        "--sequence-length", str(p["sequence_length"]),
        "--num-samples",     str(p["num_samples"]),
        "--seed",            str(p["seed"]),
    ]
    if p.get("trust_remote_code"):
        ppl_cmd.append("--trust-remote-code")

    ppl_result = subprocess.run(ppl_cmd, capture_output=True, text=True)
    if ppl_result.returncode != 0:
        raise RuntimeError(f"Perplexity failed:\n{ppl_result.stderr}")
    perplexity_report = ppl_result.stdout
    print(perplexity_report)

    # Step 3b — render model card + push (parallel with perplexity above would
    # need the report, so these run sequentially; push is I/O-bound anyway)
    print("Step 3b — rendering model card and pushing to HF...")
    template_src = cfg["model_card_template"]
    card_text = Template(template_src).render(
        model_id=model_id,
        output_repo=output_repo,
        quant=q,
        perplexity=p,
        perplexity_report=perplexity_report,
        run_date=str(date.today()),
    )
    (quant_dir / "README.md").write_text(card_text)

    api = HfApi(token=hf_token)
    api.upload_folder(
        folder_path=str(quant_dir),
        repo_id=output_repo,
        repo_type="model",
        commit_message=f"AWQ quantized model — {date.today()}",
    )

    hub_url = f"https://huggingface.co/{output_repo}"
    print(f"\nDone. Published at {hub_url}")
    return {"url": hub_url, "perplexity_report": perplexity_report}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    cfg = dict(CFG)
    cfg["model_card_template"] = (Path(__file__).parent / "model_card.md.j2").read_text()
    result = await run_pipeline(cfg)
    print(result["url"])


if __name__ == "__main__":
    asyncio.run(main())
