# Installation and running

The default installation targets local inference and learning on NVIDIA GPUs under Linux. See the [README](../README.md#features) for everyday features and [Background learning](background-learning.md#background-service) for background service configuration.

## Environment and installation

Prepare a CUDA-capable NVIDIA GPU with working drivers, which you can check with `nvidia-smi`. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), clone or download and extract the repository, and run these commands from the project root containing `pyproject.toml`:

```bash
uv sync
uv run sla
```

The project uses Python 3.12; uv selects the interpreter according to `.python-version`. Default dependencies include CUDA inference, adapter learning components, and development tools. Use `uv sync --no-dev` to exclude development tools, then `uv run --no-dev sla` to keep that installation scope. Torch/Torchvision are installed from the official CUDA 12.8 index; you must provide the GPU drivers.

On first launch, select an existing model directory or approve downloading the default Qwen3.5-4B model after confirming its source, storage requirements, and destination. The wizard estimates about 9 GB for the weights; allow additional space for dependencies, caches, and adapters. Downloads require network access. See [Chat and settings](chat-and-settings.md#first-time-setup) for conversational setup.

### Wheel installation

Install the official CUDA builds first, then the application:

```bash
uv venv --python 3.12 .sla-runtime
uv pip install --python .sla-runtime/bin/python torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .sla-runtime/bin/python ./dist/self_learning_assistant-0.1.0-py3-none-any.whl
.sla-runtime/bin/sla
```

## Model management

Register and check an existing model manually:

```bash
uv run sla model add personal --path /absolute/model/path --device cuda --load-in-4bit
uv run sla model use personal
uv run sla model check
uv run sla doctor --check-model
```

`model list` lists models; `model check` checks chat and structured output. Successful registration does not establish that a model can reliably extract memories or verify claims. Base models are stored in `base-models/` under the application data directory, and personal adapters in `adapters/`. See [Privacy and data](privacy-and-data.md#data-locations) for the actual locations.

### Local model service

Start the built-in service when you need an HTTP interface:

```bash
uv run sla serve --model-dir /absolute/model/path --device cuda --load-in-4bit --max-new-tokens 1536
```

The default address is `127.0.0.1:8091`. Options include `--port`, `--device`, `--dtype`, `--load-in-4bit`, and `--max-new-tokens`; you can also use `--model NAME` or the currently selected local model. Endpoints include `/health`, `/v1/models`, `/v1/chat/completions`, `/v1/responses`, and the owner-restricted `/v1/cancel`.

An existing local API can also be registered with `model add NAME --url URL --api-model MODEL`. The endpoint must use a loopback address and support non-streaming text chat and structured output. JSON Schema prompting is not constrained decoding. API backends do not participate in this application's parameter training. Model content is not logged by default; content diagnostics are saved only when `--trace-file FILE` is explicitly supplied.
