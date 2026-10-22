# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-file [Runpod SDK](https://docs.runpod.io/sdk/overview) (`runpod` 2.0) app. It runs a full model-quantization pipeline on a remote GPU:

1. Download a HF model and a calibration dataset.
2. AWQ-quantize the model with `mlx_lm`.
3. Run a perplexity eval.
4. Render a model card.
5. Push the result to a Hugging Face repo.

## Commands

```bash
uv sync                             # install deps (runpod 2.0) into .venv
uv run rp flash dev main.py --once  # run the @runpod.local_entrypoint once on temporary endpoints, then delete them
uv run rp flash deploy              # deploy the app
```

No lint/test tooling is configured in this repo yet.

## Configuration

- `.env` (copy from `.env.example`) must provide `HF_TOKEN`. `main.py` reads `.env` only when `runpod.is_local()` is true.
- `config.yaml` drives everything the worker does — GPU group (`flash.gpu`, e.g. `ADA_24`/`AMPERE_80`/`ADA_48_PRO`), worker count, `min_cuda_version` (format `"12.8"` or `"13.0"`), worker Python version, quantization params (`quant.*`, consumed by `mlx_lm awq`), and perplexity eval params (`perplexity.*`, consumed by `mlx_lm perplexity`). It loads once as `CFG` at module import, on the local machine only. The worker runs the module again without `config.yaml`, so the `else` branch sets placeholder values.
- `model_id`, `output_repo`, and the dataset path (`perplexity.data_path`) are expected in the dict passed to `run_pipeline` (currently that's `CFG` itself via `main()`), not hardcoded.
- The pipeline expects a `model_card.md.j2` Jinja2 template in the repo root (referenced by `main()`) — **this file does not currently exist in the repo** and must be added before `main.py` will run end-to-end.

## Architecture

Everything lives in `main.py`. Two phases, separated by a hard boundary:

1. **Module scope** (runs locally where `main.py` is invoked): loads `.env`, parses `config.yaml` into `CFG`, resolves the RunPod `GpuGroup` enum member from `config.yaml`'s `flash.gpu` string.
2. **`run_pipeline` — the `@app.queue`-decorated function**: this function executes on the remote Runpod worker, not locally. Thus it does its own imports inside the function body. The imports at module scope are for the *local* driver process only. The `dependencies=[...]` list on `@app.queue` declares the worker dependencies. Note the comment in the code: `uv_dependencies` doesn't work reliably. Thus the list holds fixed values, not values from `pyproject.toml` or `config.yaml`'s `python.dependencies`.

Pipeline steps inside `run_pipeline`, in order:
1. Parallel download of the source model and calibration dataset from HF Hub (`ThreadPoolExecutor`, 2 workers).
2. AWQ quantization via `python -m mlx_lm awq` subprocess, params from `config.yaml`'s `quant` section.
3. Perplexity evaluation via `python -m mlx_lm perplexity` subprocess, params from `quant.perplexity` section — raises on nonzero exit for both this and the quantize step.
4. Render `model_card.md.j2` with run metadata + the perplexity report, write it as `README.md` into the quantized model dir, then `HfApi.upload_folder` the whole directory to `output_repo` on the Hub.

`main()` is the local entry point (`@runpod.local_entrypoint`): it reads the Jinja template from disk, merges it into `CFG`, calls `run_pipeline.remote(...)`, and prints the resulting HF URL.

Work is staged under `/tmp/convert/` on the *worker*, in `source_model/`, `quantized_model/`, and `dataset/` subdirs.
