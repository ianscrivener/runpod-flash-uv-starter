# RunPod Flash — Hello GPU

A minimal [RunPod Flash](https://github.com/runpod/runpod-flash) worker. The `hello` endpoint runs remotely on a GPU and returns `nvidia-smi` output plus the disk usage of `/workspace`.

### Quick Start

```bash
git clone <this-repo>
cd runpod-flash-mflux-model-maker
uv sync
cp .env.example .env   # add your RUNPOD_API_KEY
uv run main.py
```

### Configuration

- `.env` — RunPod API key (see `.env.example`)
- `config.yaml` — GPU group, worker count, volume size, CUDA version
