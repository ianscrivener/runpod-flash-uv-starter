# Runpod SDK — Hello GPU

A minimal app for the [Runpod Python SDK](https://docs.runpod.io/sdk/overview) (`runpod` 2.0). The `hello` function runs on a remote GPU. It returns the `nvidia-smi` output and the disk usage of `/workspace`.

### Quick Start

```bash
git clone <this-repo>
cd runpod-flash-mflux-model-maker
uv sync
cp .env.example .env              # add your RUNPOD_API_KEY
uv run rp flash dev main.py --once
```

### Configuration

- `.env` — Runpod API key (see `.env.example`)
- `config.yaml` — GPU group, worker count, volume size, CUDA version

The worker runs `main.py` again, but it does not get `.env` or `config.yaml`. Thus `main.py` reads these files only when `runpod.is_local()` is true.
