# Yllia Safety Classifier

## Local Gemma 4 safety adapter for Polish mental-health AI workflows

Yllia Safety Classifier is a **local-first safety classification API and browser demo** for Polish mental-health AI systems. Public repository: [github.com/shivihs/yllia-safety-classifier](https://github.com/shivihs/yllia-safety-classifier.git). It runs a Gemma 4 base model with a fine-tuned LoRA/PEFT adapter and classifies user messages before a downstream chatbot, agent, intake form, or RAG system decides how to respond.[1] [3]

The project is designed as a **reproducible local demo** rather than a permanently hosted GPU endpoint. This is intentional: the target use case is privacy-preserving, on-premise, edge, or air-gapped deployment, where sensitive user content should remain under local control.

> **Helpful AI answers questions. Safe AI first recognizes risk.**

## What this repository contains

This repository exposes a Dockerized FastAPI service with an HTML demo UI. The backend loads a Gemma 4 model and a local LoRA adapter from the `final/` directory. Polish input is sent directly to the classifier. English input can be translated locally into Polish using offline translation models, so jurors and reviewers can test the demo without external translation APIs.

| Component | File | Purpose |
|---|---|---|
| FastAPI backend | `app.py` | Loads the model and adapter, exposes the API, performs classification and optional local translation. |
| Web demo | `index.html` | Browser interface for testing Polish or English messages through `/api/demo`. |
| Docker image | `Dockerfile` | Builds the CUDA/NVIDIA-based runtime with Unsloth, Transformers, PEFT, and API dependencies. |
| Runtime config | `docker-compose.yml` | Starts the API on port `11434`, mounts the adapter, Hugging Face cache, and GPU resources. |
| Python dependencies | `requirements.txt` | Minimal API and translation dependencies used by the container. |
| Adapter directory | `final/` | Expected local directory containing the trained LoRA/PEFT adapter files. |

## Safety taxonomy

The classifier returns one of five safety labels. These labels are intended for **routing**, not for diagnosis or medical decision-making.

| Label | Meaning | Suggested downstream behavior |
|---|---|---|
| `OK` | Relevant, non-sensitive administrative or service-related message. | Route to normal assistant or intake workflow. |
| `OFFTOPIC` | Message unrelated to the intended mental-health or service context. | Redirect politely to the supported scope. |
| `MEDICAL_SENSITIVE` | Medication, diagnosis, symptoms, treatment decisions, side effects, or other clinically sensitive content. | Do not provide automated medical advice; route to supervised medical information flow. |
| `CRISIS` | Potential acute crisis, self-harm, suicidal ideation, severe psychological distress, or similar urgent-risk signal. | Trigger crisis or urgent-help protocol; do not treat as ordinary chat. |
| `ATTACK` | Prompt injection, manipulation, spam, abusive language, or attempts to bypass rules. | Ignore malicious instructions and apply the system’s refusal or hardening policy. |

## Architecture

The system separates **safety classification** from downstream answer generation. The adapter does not answer the user, diagnose, or replace clinical judgment. It produces a structured JSON decision that can be consumed by another system.

```text
user message
   ↓
optional local EN → PL translation
   ↓
Gemma 4 + LoRA safety adapter
   ↓
structured JSON classification
   ↓
risk-based routing policy
   ↓
assistant / cautious medical flow / crisis protocol / refusal / human review
```

This separation is the main design choice. A RAG system may help an assistant retrieve relevant knowledge, but retrieval alone is not a guardrail. Yllia Safety Classifier is meant to run **before** the assistant response is generated.

## Requirements

The demo is GPU-oriented because it loads a Gemma 4 model with an adapter. It is packaged for Docker Compose with NVIDIA GPU access.[4] [5]

| Requirement | Notes |
|---|---|
| Operating system | Linux host recommended for NVIDIA container support. |
| Docker | Docker Engine with Compose plugin. |
| GPU | NVIDIA GPU recommended. For Gemma 4 E4B, 4-bit loading is enabled by default. |
| NVIDIA runtime | NVIDIA Container Toolkit must be installed and working. |
| Adapter files | A local `final/` directory containing the trained LoRA/PEFT adapter, including `adapter_config.json`. |
| Model access | If the base model requires authentication, configure Hugging Face access on the host or pass a token to the container. |
| Disk/network | First run downloads the base model and offline translation models into the Hugging Face cache volume through the Hugging Face/Transformers ecosystem.[2] |

If you do not have a permanent cloud GPU, this repository can still serve as the competition demo by providing a reproducible local run path. Reviewers can run the same container on any compatible GPU machine.

## Repository layout

A typical repository should look like this:

```text
.
├── app.py
├── docker-compose.yml
├── Dockerfile
├── index.html
├── README.md
├── requirements.txt
└── final/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── ...
```

The `final/` directory is mounted read-only into the container at `/models/final` by `docker-compose.yml`.

```yaml
volumes:
  - ./final:/models/final:ro
  - hf-cache:/root/.cache/huggingface
```

If your adapter is stored elsewhere, change the left side of the volume mapping.

## Quick start

Clone the repository, place the trained adapter in `final/`, and start the service with Docker Compose.

```bash
git clone https://github.com/shivihs/yllia-safety-classifier.git
cd yllia-safety-classifier

# The final/ directory must contain adapter_config.json and adapter weights.
ls final

docker compose down
docker compose build --no-cache
docker compose up
```

When the service is ready, open the local demo UI:

```text
http://localhost:11434/
```

The first run may take longer because the base model and translation models are downloaded into the `hf-cache` Docker volume. Later runs reuse the cache.

## Hugging Face access

If the configured base model requires Hugging Face authentication, make sure your environment has access before starting the container. One practical approach is to log in on the host and mount or reuse the Hugging Face cache. Another approach is to add an environment variable such as `HF_TOKEN` to `docker-compose.yml`.

```yaml
environment:
  HF_TOKEN: ${HF_TOKEN}
```

Then start the service with:

```bash
export HF_TOKEN=<your-hugging-face-token>
docker compose up
```

Only use tokens that are appropriate for your local environment. Do not commit secrets to the repository.

## Demo flow

The browser UI is served by `GET /` and sends requests to `/api/demo`. It supports three language modes: `auto`, `pl`, and `en`.

| Step | Polish input | English input |
|---|---|---|
| 1 | User message is sent directly to the classifier. | User message is translated locally from English to Polish. |
| 2 | Gemma 4 + adapter classifies the Polish message. | Gemma 4 + adapter classifies the translated Polish message. |
| 3 | API returns class, rationale, and routing recommendation. | API also attempts to translate the rationale back to English. |

The English path is a demo bridge. The classifier itself is **Polish-native** and was designed around Polish mental-health safety data.

## API endpoints

The service exposes a minimal Ollama-compatible API shape for local integrations, plus a richer demo endpoint.

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | `GET` | Web demo UI. |
| `/health` | `GET` | Runtime status, device information, model loading state, and translation configuration. |
| `/api/tags` | `GET` | Minimal model metadata endpoint. |
| `/api/chat` | `POST` | Minimal Ollama-compatible chat endpoint for Polish-native classification. |
| `/api/generate` | `POST` | Minimal Ollama-compatible generate endpoint for Polish-native classification. |
| `/api/demo` | `POST` | Demo endpoint with Polish/English input handling, parsed class, rationale, and routing guidance. |

## API examples

### Health check

```bash
curl http://localhost:11434/health
```

Example response:

```json
{
  "status": "ok",
  "model": "gemma4-e4b-safety-reasoning",
  "adapter_path": "/models/final",
  "loaded": true,
  "device": "cuda",
  "cuda_available": true,
  "translation_mode": "offline"
}
```

### Polish-native chat endpoint

```bash
curl http://localhost:11434/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemma4-e4b-safety-reasoning",
    "stream": false,
    "messages": [
      {
        "role": "user",
        "content": "Czy mogę sam zwiększyć dawkę sertraliny?"
      }
    ]
  }'
```

The model response is expected to contain JSON similar to:

```json
{
  "kategoria": "MEDICAL_SENSITIVE",
  "uzasadnienie": "Pytanie dotyczy samodzielnej zmiany dawkowania leku, co wymaga konsultacji medycznej."
}
```

### English demo endpoint with local translation

```bash
curl http://localhost:11434/api/demo \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Can I increase my medication dose by myself?",
    "language": "en"
  }'
```

Example response shape:

```json
{
  "input_language": "en",
  "original_text": "Can I increase my medication dose by myself?",
  "polish_text": "Czy mogę samodzielnie zwiększyć dawkę leku?",
  "model_result": {
    "kategoria": "MEDICAL_SENSITIVE",
    "uzasadnienie": "Pytanie dotyczy samodzielnej modyfikacji dawkowania leku."
  },
  "category_en": "Medical-sensitive",
  "rationale_en": "The question concerns independently modifying the dosage of medication.",
  "route_pl": "Nie udzielać automatycznej porady medycznej; przekierować do lekarza lub konsultacji.",
  "route_en": "Do not provide automated medical advice; route to a clinician or consultation.",
  "timing": {
    "total_duration_ns": 1423887000,
    "model_eval_duration_ns": 900000000
  }
}
```

## Configuration

Most runtime settings are configured in `docker-compose.yml` through environment variables.

| Variable | Default in compose | Description |
|---|---|---|
| `MODEL_NAME` | `gemma4-e4b-safety-reasoning` | Local model name exposed by API responses. |
| `BASE_MODEL` | `google/gemma-4-E4B-it` | Hugging Face base model identifier. |
| `ADAPTER_PATH` | `/models/final` | Adapter path inside the container. |
| `CHAT_TEMPLATE` | `gemma-4` | Chat template used by Unsloth. |
| `MAX_SEQ_LENGTH` | `1024` | Maximum sequence length for model input. |
| `MAX_NEW_TOKENS` | `128` | Maximum generated tokens for the JSON response. |
| `TEMPERATURE` | `0.0` | Deterministic generation by default. |
| `TOP_P` | `0.9` | Top-p value used only when sampling is enabled. |
| `LOAD_IN_4BIT` | `true` | Enables 4-bit loading, useful for lower VRAM GPUs. |
| `LOAD_IN_8BIT` | `false` | Optional 8-bit loading; do not enable together with 4-bit. |
| `LOAD_ON_STARTUP` | `1` | Loads the model during API startup. |
| `TRANSLATION_MODE` | `offline` | Enables offline EN/PL translation for the demo. Use `none` to disable. |
| `TRANSLATION_DEVICE` | `cpu` | Device for translation models. CPU keeps GPU memory for the classifier. |
| `EN_PL_MODEL` | `Helsinki-NLP/opus-mt-en-sla` | Offline English-to-Slavic translation model with Polish target prefix. |
| `PL_EN_MODEL` | `Helsinki-NLP/opus-mt-pl-en` | Offline Polish-to-English translation model. |
| `LOAD_TRANSLATION_ON_STARTUP` | `0` | Loads translation lazily by default. Set to `1` to preload it. |

## VRAM and model-size notes

For Gemma 4 E4B on a 16 GB VRAM GPU, the compose file enables 4-bit loading by default:

```yaml
LOAD_IN_4BIT: "true"
LOAD_IN_8BIT: "false"
```

If you have more VRAM and want to test higher precision, you can disable 4-bit loading. If you have less memory, consider using a smaller compatible adapter and changing `BASE_MODEL` accordingly.

## Offline and cache behavior

The repository does not use Google Translate, DeepL, OpenAI, or any other external inference API for translation; translation is handled locally through Hugging Face-compatible models.[2] The first run downloads required models into the Docker volume named `hf-cache`. After the cache is populated, later runs reuse the downloaded files.

Do not enable a strict offline mode until all required assets have already been cached. If you need air-gapped execution, prepare the Hugging Face cache on a connected machine first and then move it into the target environment.

## Troubleshooting

| Problem | Likely cause | Suggested fix |
|---|---|---|
| `adapter_config.json not found` | `final/` is missing or mounted incorrectly. | Confirm that `./final:/models/final:ro` points to the adapter directory. |
| CUDA is not visible in `/health` | Docker cannot access the NVIDIA GPU. | Check NVIDIA drivers, NVIDIA Container Toolkit, and Compose GPU configuration. |
| Model download fails | Missing Hugging Face access or network issue. | Accept model terms if required, configure `HF_TOKEN`, and rerun. |
| Container exits during startup | VRAM shortage or dependency/runtime mismatch. | Keep `LOAD_IN_4BIT=true`, reduce model size, or inspect container logs. |
| English demo is slow on first request | Translation models are loaded lazily. | Set `LOAD_TRANSLATION_ON_STARTUP=1` or wait for first lazy load. |
| `/api/demo` returns translation error | Translation model not downloaded or unavailable. | Check network/cache and verify `TRANSLATION_MODE=offline`. |

Useful diagnostic commands:

```bash
docker compose logs -f
curl http://localhost:11434/health
docker compose down
```

## Why there is no permanently hosted GPU endpoint

This project is designed for **local and private deployment**, not as a cloud-only chatbot. Mental-health messages can be highly sensitive, and the intended use cases include local clinics, schools, on-premise systems, edge devices, and controlled environments where data should remain inside the organization’s infrastructure.

For that reason, the competition demo is provided as a **reproducible Docker-based local demo**. It can be run on any compatible GPU machine and reviewed through the browser UI or API endpoints.

## Medical and safety disclaimer

Yllia Safety Classifier is a research and engineering prototype for AI safety routing. It is **not** a medical device, diagnostic system, therapy tool, crisis intervention service, or replacement for a clinician. The classifier output should be treated as a routing signal for downstream safety policies and human-supervised workflows.

If this component is used in a real system, crisis handling, medical escalation, human review, logging, privacy controls, and jurisdiction-specific compliance must be implemented separately.

## References

[1]: https://www.kaggle.com/competitions/gemma-4-good-hackathon/overview "The Gemma 4 Good Hackathon — Kaggle"
[2]: https://huggingface.co/docs/transformers/index "Transformers documentation — Hugging Face"
[3]: https://huggingface.co/docs/peft/index "PEFT documentation — Hugging Face"
[4]: https://docs.docker.com/compose/ "Docker Compose documentation"
[5]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/ "NVIDIA Container Toolkit documentation"
