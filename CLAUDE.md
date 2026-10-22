# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-file [RunPod Flash](https://github.com/runpod/runpod-flash) worker that runs a full model-quantization pipeline on a remote GPU: download a HF model + calibration dataset, AWQ-quantize it with `mlx_lm`, run a perplexity eval, render a model card, and push the result to a Hugging Face repo.

## Commands

```bash
uv sync              # install deps (runpod-flash) into .venv
uv run main.py        # trigger the pipeline — submits the @Endpoint job to RunPod Flash
```

No lint/test tooling is configured in this repo yet.

## Configuration

- `.env` (copy from `.env.example`) must provide `HF_TOKEN`; `main.py` reads it at import time via `os.environ["HF_TOKEN"]` and fails fast if absent.
- `config.yaml` drives everything the worker does — GPU group (`flash.gpu`, e.g. `ADA_24`/`AMPERE_80`/`ADA_48`), worker Python version, quantization params (`quant.*`, consumed by `mlx_lm awq`), and perplexity eval params (`perplexity.*`, consumed by `mlx_lm perplexity`). It's loaded once as `CFG` at module import.
- `model_id`, `output_repo`, and the dataset path (`perplexity.data_path`) are expected in the dict passed to `run_pipeline` (currently that's `CFG` itself via `main()`), not hardcoded.
- The pipeline expects a `model_card.md.j2` Jinja2 template in the repo root (referenced by `main()`) — **this file does not currently exist in the repo** and must be added before `main.py` will run end-to-end.

## Architecture

Everything lives in `main.py`. Two phases, separated by a hard boundary:

1. **Module scope** (runs locally where `main.py` is invoked): loads `.env`, parses `config.yaml` into `CFG`, resolves the RunPod `GpuGroup` enum member from `config.yaml`'s `flash.gpu` string.
2. **`run_pipeline` — the `@Endpoint`-decorated function**: this is what actually executes on the remote RunPod Flash worker, not locally. Because of that, it does its own imports inside the function body (the imports at module scope are for the *local* driver process only, not the worker's environment). Worker dependencies are declared explicitly via the `dependencies=[...]` list on `@Endpoint` — note the comment in the code: `uv_dependencies` doesn't work reliably, so deps are hardcoded there rather than derived from `pyproject.toml`/`config.yaml`'s `python.dependencies`.

Pipeline steps inside `run_pipeline`, in order:
1. Parallel download of the source model and calibration dataset from HF Hub (`ThreadPoolExecutor`, 2 workers).
2. AWQ quantization via `python -m mlx_lm awq` subprocess, params from `config.yaml`'s `quant` section.
3. Perplexity evaluation via `python -m mlx_lm perplexity` subprocess, params from `quant.perplexity` section — raises on nonzero exit for both this and the quantize step.
4. Render `model_card.md.j2` with run metadata + the perplexity report, write it as `README.md` into the quantized model dir, then `HfApi.upload_folder` the whole directory to `output_repo` on the Hub.

`main()` is the local entry point: it reads the Jinja template from disk, merges it into `CFG`, awaits `run_pipeline`, and prints the resulting HF URL.

Work is staged under `/tmp/convert/` on the *worker*, in `source_model/`, `quantized_model/`, and `dataset/` subdirs.
