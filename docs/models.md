# Model compatibility

GeoAI-VLM talks to vision-language models through three model-agnostic
backends (Transformers, vLLM, any OpenAI-compatible server) and embeds with
Qwen3-VL-Embedding or any CLIP-family dual encoder. This page separates what
has **actually been run** from what is only **expected** to work. Nothing in
the second table has been run; please do not read it as a claim.

A passing run means the software path works for that model: it loads, the
chat template takes our messages, an answer comes back and is parsed and
recorded with provenance. It says nothing about description quality.

## Check a model yourself

```bash
geoai-vlm check-model HuggingFaceTB/SmolVLM2-256M-Video-Instruct
geoai-vlm check-model Qwen/Qwen3-VL-2B-Instruct --template geoai --max-new-tokens 1024 --json
geoai-vlm check-model my-model --backend openai --base-url http://localhost:8000/v1
```

The check describes one synthetic drawing (no third-party imagery) and
reports load time, whether a chat template exists, whether it accepts a system
turn (or the system prompt had to be prepended), whether the answer parsed as
JSON, schema issues, generation time and peak memory. From Python:
`geoai_vlm.check_model(...)` returns the same report.

## Tested: models actually run

Date 2026-10-10. Hardware: 4 vCPU Intel Xeon @ 2.10 GHz, 15 GB RAM, **no
GPU** (CPU only, float32). Python 3.12, torch 2.14.1+cpu, huggingface_hub
2.2.0 (0.36.2 with transformers 4.57.3). Backend: Transformers. Image: the
synthetic drawing. Greedy decoding. Times are wall-clock seconds on this
machine; RSS is the process peak.

### Description models

| Model | Family | Revision | transformers | Template (max tokens) | Result | Load s | Generate s | Peak RSS MB | Notes |
|---|---|---|---|---|---|---|---|---|---|
| `HuggingFaceTB/SmolVLM-256M-Instruct` | Idefics3 | `7e3e67ed` | 5.19.0 | simple (256) | partial | 4.9 | 23.6 | 2163 | Loads and generates; the answer is not valid JSON (the model falls into a repetition loop). |
| `HuggingFaceTB/SmolVLM2-256M-Video-Instruct` | SmolVLM2 | `067788b1` | 5.19.0 | simple (256) | ok | 5.4 | 23.2 | 1669 | Needs `pip install num2words` (required by its processor). 2 schema issues: tags returned as objects, not strings. |
| `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` | LLaVA-OneVision | `74dd0bf8` | 5.19.0 | simple (256) | ok | 6.9 | 23.1 | 5740 | |
| `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` | LLaVA-OneVision | `74dd0bf8` | 4.57.3 | simple (256) | ok | 7.6 | 14.9 | 5674 | |
| `OpenGVLab/InternVL3-1B-hf` | InternVL3 | `014c0583` | 5.19.0 | simple (256) | ok | 7.2 | 35.0 | 6004 | Native (`-hf`) checkpoint; no remote code. |
| `Qwen/Qwen3-VL-2B-Instruct` (default) | Qwen3-VL | `89644892` | 5.19.0 | simple (256) | ok | 9.3 | 31.2 | 12055 | |
| `Qwen/Qwen3-VL-2B-Instruct` (default) | Qwen3-VL | `89644892` | 4.57.3 | simple (256) | ok | 10.2 | 32.1 | 12333 | |
| `Qwen/Qwen3-VL-2B-Instruct` (default) | Qwen3-VL | `89644892` | 5.19.0 | geoai (1024) | ok | 10.2 | 187.0 | 12155 | Full GeoAI schema: parsed, 0 schema issues. |

The `active_mobility_audit_v1` template was run once with
`Qwen/Qwen3-VL-2B-Instruct` (`89644892`, transformers 5.19.0, 1024 tokens) on
the synthetic drawing, through the demo service: all 19 items came back with
valid states, confidences and evidence (0 normalisation issues). The GeoAI
description and the audit together took 482 s on this CPU. The answers are
plausible for the drawing but not all correct (the model called the road's
centre line part of a sidewalk) -- a reminder that observations need
reference checks.

Every template above accepted a system turn (`system_role: supported`), so no
fallback to prepending was needed for these models. Two-image batched
generation (different image sizes, left padding) was also run with
SmolVLM-256M on transformers 5.19.0 and 4.57.3: outputs came back in input
order with provenance recorded (`tests/test_real_models.py`).

### Through a real OpenAI-compatible server

| Server | Model | Result | Notes |
|---|---|---|---|
| `transformers serve` (transformers 5.19.0, CPU) | `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` | ok | `check-model --backend openai`: JSON parsed, 0 schema issues, 20.5 s. `describe()` of two images with `max_concurrency=2` and `structured_output=True`: both parsed. This server **accepts `response_format` but ignores it** (it logs "Ignoring unsupported fields"), which is why such records say `decoding_mode="json_schema_requested"` rather than claiming enforcement. |

### Research demo

`geoai-vlm`'s demo service and Gradio interface were run locally with
`Qwen/Qwen3-VL-2B-Instruct` (description, audit and text chat) and a 300-scene
index built by `examples/build_demo_index.py` from the published dataset
(SigLIP 2 text embeddings). Through the interface's API: nearest scenes to a
coordinate were returned with a map; a question about trees was answered
with four image-id citations, all among the retrieved records; a question
about pedestrian shopping streets was declined (the model found no support in
the retrieved records). Retrieval itself was weak: SigLIP 2 gave every
narrative a similarity of about 0.88-0.89 to the question, so it hardly
discriminates between descriptions written in the same style.

### Embedding and segmentation models

| Model | Use | Revision | Result | Notes |
|---|---|---|---|---|
| `google/siglip2-base-patch16-224` | `ImageEmbedder(backend="clip")` | `75de2d55` | ok | 768-d, unit-norm; batched equals one-by-one; each of 3 synthetic images ranks its matching caption first. |
| `openai/clip-vit-base-patch32` | `ImageEmbedder(backend="clip")` | `3d74acf9` | ok | 512-d, unit-norm; same caption-ranking sanity check passed. |
| `nvidia/segformer-b0-finetuned-cityscapes-1024-1024` | `SemanticSegmenter` | `21b3847f` | ok | Labels match `cityscapes-19@1`; fractions sum to 1 on a synthetic image. |

## Not run: expected to be compatible

These families are registered as image-text-to-text models in current
transformers releases or are documented as supported by vLLM, so the
model-agnostic path *should* work. **None of them was run for this release.**

| Family / example | Status | Why not run / what to watch |
|---|---|---|
| Gemma 3 (`google/gemma-3-4b-it`) | untested | Gated: needs licence acceptance and an HF token. |
| Gemma 3n | untested | Gated. |
| Qwen2.5-VL (`Qwen/Qwen2.5-VL-3B-Instruct`) | untested | Not run for time; same chat path as Qwen3-VL. |
| Larger SmolVLM2 / Idefics3 (500M, 2.2B, 8B) | untested | |
| Larger InternVL3 `-hf` checkpoints | untested | Non-`-hf` InternVL checkpoints need `trust_remote_code=True`. |
| Larger LLaVA-OneVision | untested | |
| Llama 3.2 Vision, Mistral / Pixtral vision models | untested | Some templates reject a system turn: `system_prompt_mode="auto"` should fall back to prepending. Gated in part. |
| **vLLM backend (any model)** | untested with a real model | No GPU here. `LLM.chat`, `SamplingParams(structured_outputs=...)` and `image_pil` parts were checked against the vLLM 0.13.0 and 0.31.0 sources and exercised with a fake `vllm` module. |
| **Other OpenAI-compatible servers** (vLLM server, TGI, SGLang, Ollama, LM Studio, hosted APIs) | untested | Only `transformers serve` was run. Retries, timeouts, `response_format` and system-role fallbacks and key redaction were tested against a local fake server. |
| 4-bit / 8-bit loading (`quantization=`) | untested | Needs a CUDA GPU and bitsandbytes. |
| Qwen3-VL-Embedding backends | not re-run | Unchanged code path apart from `trust_remote_code` (now off by default; the checkpoints load with native classes). |

## Library version notes

Verified by reading the installed sources:

* **transformers 5** removed `AutoModelForVision2Seq` (it is an
  `ImportError` on 5.19.0), so the 0.3 Transformers backend could not load
  any model on a current install. The backend now uses
  `AutoModelForImageTextToText` (present since 4.46), with a controlled
  fallback to Vision2Seq on older installs.
* `ProcessorMixin.apply_chat_template` takes processor arguments as a
  `processor_kwargs={...}` dict on transformers 5 and as keyword arguments on
  4.x. The backend detects the signature; `padding` and `padding_side` are
  accepted by both.
* `from_pretrained(dtype=...)` exists from 4.56; `torch_dtype` is deprecated.
* `CLIPModel` / `SiglipModel` `get_image_features` and `get_text_features`
  return a tensor on 4.x and a `BaseModelOutputWithPooling` (features in
  `pooler_output`) on 5.x. The CLIP backend handles both.
* **vLLM 0.13.0 and 0.31.0** share the `LLM.chat(messages, sampling_params,
  use_tqdm, chat_template_content_format, ...)` signature, the
  `SamplingParams(structured_outputs=StructuredOutputsParams(json=...))` API
  and the `image_pil` content part. `max_model_len`, `max_num_seqs` and
  `limit_mm_per_prompt` go through `EngineArgs`. vLLM 0.31 requires
  transformers `>=5.10.4,<5.18`; vLLM 0.13-0.19 require transformers `<5`.
* The SmolVLM / SmolVLM2 processors import `num2words`.
