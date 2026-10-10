# -*- coding: utf-8 -*-
"""
Chat Messages for GeoAI-VLM
===========================
Model-agnostic chat construction shared by every description backend.

A vision-language model is asked the same thing whichever library runs it: a
system instruction, an image and a user instruction. What differs is how each
library expects the image to be attached and whether the model's chat template
accepts a ``system`` turn at all. :func:`build_chat_messages` produces the one
canonical message list; each backend only chooses the image style.

Some chat templates reject a system turn (they raise), and some silently drop
it. Either way the model would never see the output schema, so
:func:`resolve_system_prompt_mode` probes the template once and, when needed,
moves the system text to the start of the first user turn instead. Which of
the two actually happened is recorded per description as
``system_prompt_mode_effective``.
"""

from __future__ import annotations

import base64
import io
import logging
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from PIL import Image


__all__ = [
    "SYSTEM_PROMPT_MODES",
    "GenerationOutput",
    "build_chat_messages",
    "resolve_system_prompt_mode",
    "load_image",
    "image_to_data_url",
    "make_synthetic_street_image",
]

logger = logging.getLogger(__name__)

#: Accepted values for ``system_prompt_mode``.
#:
#: ``"auto"``     use a system turn when the chat template accepts it, else
#:                prepend the system text to the first user text
#: ``"system"``   always send a system turn
#: ``"prepend"``  never send a system turn; prepend its text to the user text
SYSTEM_PROMPT_MODES = ("auto", "system", "prepend")

#: Effective modes recorded on each description.
EFFECTIVE_MODES = ("system", "prepend", "none")

#: Image attachment styles understood by :func:`build_chat_messages`.
IMAGE_STYLES = ("hf", "openai", "vllm_pil")

# A sentinel that cannot plausibly occur in a real chat template. Used to
# detect templates that accept a system turn but drop its text.
_PROBE_SENTINEL = "geoai-vlm-system-probe-5c1f8e"


@dataclass
class GenerationOutput:
    """One backend response, with how it was produced.

    Attributes:
        text: The raw text the model returned (empty when generation failed).
        error: Set when no response could be generated for this item. The
            description record is then marked as failed and retried on resume.
        decoding_mode: ``"json_schema"`` when decoding was constrained by a
            JSON schema locally (vLLM), ``"json_schema_requested"`` when the
            schema was sent to a server that accepted it (enforcement is up to
            the server), ``"unconstrained"`` otherwise.
        system_prompt_mode: ``"system"``, ``"prepend"`` or ``"none"`` -- how the
            system prompt actually reached the model.
    """

    text: str
    error: Optional[str] = None
    decoding_mode: Optional[str] = None
    system_prompt_mode: Optional[str] = None


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
ImageInput = Union[str, Path, bytes, bytearray, Image.Image]


def _is_url(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(("http://", "https://"))


def _is_data_url(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("data:")


def load_image(image: ImageInput) -> Image.Image:
    """Return *image* as an RGB :class:`PIL.Image.Image`.

    Accepts a local path, a PIL image, raw bytes or a ``data:`` URL. Remote
    ``http(s)`` URLs are deliberately not fetched here: backends that can
    accept a URL pass it through untouched.
    """
    if isinstance(image, Image.Image):
        return image if image.mode == "RGB" else image.convert("RGB")
    if isinstance(image, (bytes, bytearray)):
        with Image.open(io.BytesIO(bytes(image))) as img:
            return img.convert("RGB")
    if _is_data_url(image):
        try:
            payload = image.split(",", 1)[1]
        except IndexError as exc:
            raise ValueError("malformed data URL") from exc
        return load_image(base64.b64decode(payload))
    if _is_url(image):
        raise ValueError(
            "load_image does not download remote URLs; pass a local path, "
            "a PIL image or bytes"
        )
    with Image.open(Path(image)) as img:
        return img.convert("RGB")


_PASSTHROUGH_MIME = {"image/jpeg", "image/png", "image/webp", "image/gif"}


def image_to_data_url(
    image: ImageInput,
    max_side: Optional[int] = None,
    image_format: str = "JPEG",
    quality: int = 90,
) -> str:
    """Encode *image* as a base64 ``data:`` URL.

    A local JPEG/PNG/WebP file that needs no resizing is sent byte-for-byte, so
    the server sees exactly the file on disk. Anything else is re-encoded as
    *image_format*. ``max_side`` downsizes the longer side (aspect preserved).
    """
    if _is_data_url(image) and max_side is None:
        return image  # already encoded

    if isinstance(image, (str, Path)) and not _is_url(image) and not _is_data_url(image):
        path = Path(image)
        mime = mimetypes.guess_type(path.name)[0]
        if max_side is None and mime in _PASSTHROUGH_MIME:
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:{mime};base64,{encoded}"

    pil = load_image(image)
    if max_side is not None and max(pil.size) > max_side:
        scale = max_side / float(max(pil.size))
        new_size = (max(1, round(pil.width * scale)), max(1, round(pil.height * scale)))
        pil = pil.resize(new_size, Image.Resampling.BICUBIC)

    buffer = io.BytesIO()
    fmt = image_format.upper()
    save_kwargs: Dict[str, Any] = {}
    if fmt in ("JPEG", "JPG"):
        fmt = "JPEG"
        save_kwargs["quality"] = quality
    pil.save(buffer, format=fmt, **save_kwargs)
    mime = Image.MIME.get(fmt, f"image/{fmt.lower()}")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _image_part(image: Any, image_style: str) -> Dict[str, Any]:
    if image_style == "hf":
        if _is_url(image):
            return {"type": "image", "url": image}
        return {"type": "image", "image": load_image(image)}
    if image_style == "vllm_pil":
        if _is_url(image):
            return {"type": "image_url", "image_url": {"url": image}}
        return {"type": "image_pil", "image_pil": load_image(image)}
    if image_style == "openai":
        url = image if _is_url(image) else image_to_data_url(image)
        return {"type": "image_url", "image_url": {"url": url}}
    raise ValueError(f"image_style must be one of {IMAGE_STYLES}, got {image_style!r}")


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------
def build_chat_messages(
    user_prompt: str,
    image: Optional[ImageInput] = None,
    system_prompt: Optional[str] = None,
    *,
    mode: str = "system",
    image_style: str = "hf",
    system_content: str = "parts",
) -> List[Dict[str, Any]]:
    """Build one model-agnostic chat for a single image.

    Args:
        user_prompt: The user instruction.
        image: Path, PIL image, bytes, ``data:`` URL or remote URL. ``None``
            builds a text-only chat.
        system_prompt: The system instruction, if any.
        mode: ``"system"`` sends a system turn; ``"prepend"`` places the
            system text before the user text in the first user turn.
            ``"auto"`` means "try the system turn" and is treated as
            ``"system"`` here -- backends resolve it with
            :func:`resolve_system_prompt_mode` before calling this.
        image_style: How the image is attached. ``"hf"`` for Hugging Face
            processors, ``"vllm_pil"`` for vLLM's in-memory image part,
            ``"openai"`` for an OpenAI-compatible ``image_url`` part.
        system_content: ``"parts"`` sends the system text as a list of content
            parts (what multimodal processor templates expect); ``"string"``
            sends a plain string (the most widely accepted form on
            OpenAI-compatible servers).

    Returns:
        A list of ``{"role": ..., "content": ...}`` messages. The image, when
        given, precedes the text in the user turn.
    """
    if mode not in SYSTEM_PROMPT_MODES:
        raise ValueError(f"mode must be one of {SYSTEM_PROMPT_MODES}, got {mode!r}")
    if system_content not in ("parts", "string"):
        raise ValueError("system_content must be 'parts' or 'string'")

    has_system = bool(system_prompt and system_prompt.strip())
    text = user_prompt
    if has_system and mode == "prepend":
        text = f"{system_prompt.strip()}\n\n{user_prompt}"

    user_content: List[Dict[str, Any]] = []
    if image is not None:
        user_content.append(_image_part(image, image_style))
    user_content.append({"type": "text", "text": text})

    messages: List[Dict[str, Any]] = []
    if has_system and mode in ("system", "auto"):
        if system_content == "parts":
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            )
        else:
            messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    return messages


def resolve_system_prompt_mode(
    requested: str,
    system_prompt: Optional[str],
    render: Optional[Callable[[List[Dict[str, Any]]], Any]] = None,
    *,
    image_style: str = "hf",
) -> Tuple[str, Optional[str]]:
    """Decide how the system prompt will actually reach the model.

    Args:
        requested: ``"auto"``, ``"system"`` or ``"prepend"``.
        system_prompt: The system text. Empty or ``None`` yields ``"none"``.
        render: A callable that renders a message list with the model's chat
            template (without tokenising or loading images). Only used for
            ``"auto"``.
        image_style: Style of the placeholder image part used in the probe.

    Returns:
        ``(effective_mode, reason)``. ``effective_mode`` is ``"system"``,
        ``"prepend"`` or ``"none"``; ``reason`` explains a fallback (or says
        that ``"auto"`` could not be verified), and is ``None`` otherwise.

    For ``"auto"`` the template is rendered once with a sentinel system text.
    If rendering raises, or the sentinel is missing from the result (the
    template silently dropped the system turn), the effective mode is
    ``"prepend"``. Without a *render* callable ``"auto"`` resolves to
    ``"system"`` and the backend falls back on a template error at call time.
    """
    if requested not in SYSTEM_PROMPT_MODES:
        raise ValueError(
            f"system_prompt_mode must be one of {SYSTEM_PROMPT_MODES}, got {requested!r}"
        )
    if not (system_prompt and system_prompt.strip()):
        return "none", None
    if requested != "auto":
        return requested, None
    if render is None:
        return "system", "auto: chat template not inspectable; system turn attempted"

    placeholder = {"type": "image"} if image_style == "hf" else None
    probe = [
        {"role": "system", "content": [{"type": "text", "text": _PROBE_SENTINEL}]},
        {
            "role": "user",
            "content": ([placeholder] if placeholder else [])
            + [{"type": "text", "text": "probe"}],
        },
    ]
    try:
        rendered = render(probe)
    except Exception as exc:  # the template refused a system turn
        reason = f"chat template rejected the system role ({type(exc).__name__}: {exc})"
        logger.warning("system prompt will be prepended to the user turn: %s", reason)
        return "prepend", reason

    if _PROBE_SENTINEL not in str(rendered):
        reason = "chat template accepted but dropped the system turn"
        logger.warning("system prompt will be prepended to the user turn: %s", reason)
        return "prepend", reason
    return "system", None


# ---------------------------------------------------------------------------
# A synthetic test image
# ---------------------------------------------------------------------------
def make_synthetic_street_image(width: int = 384, height: int = 256) -> Image.Image:
    """Draw a simple, clearly synthetic street scene.

    Used for dry runs (:func:`geoai_vlm.models.check_model`) and tests so that
    no third-party photograph is needed or redistributed. It is a drawing of
    sky, two building blocks, a road, a sidewalk and a tree -- not imagery of
    any real place.
    """
    from PIL import ImageDraw

    img = Image.new("RGB", (width, height), (135, 190, 235))  # sky
    draw = ImageDraw.Draw(img)
    horizon = int(height * 0.45)
    draw.rectangle([0, int(height * 0.12), int(width * 0.32), horizon + 40], fill=(170, 120, 90))
    draw.rectangle([int(width * 0.68), int(height * 0.08), width, horizon + 40], fill=(200, 190, 170))
    for x in range(int(width * 0.04), int(width * 0.28), int(width * 0.07)):
        draw.rectangle([x, int(height * 0.2), x + int(width * 0.04), int(height * 0.3)], fill=(60, 80, 110))
    draw.polygon(
        [(int(width * 0.35), horizon), (int(width * 0.65), horizon), (width, height), (0, height)],
        fill=(80, 80, 85),
    )  # road
    draw.polygon(
        [(int(width * 0.65), horizon), (int(width * 0.72), horizon), (width, int(height * 0.8)), (width, height)],
        fill=(180, 180, 175),
    )  # sidewalk
    for y in range(horizon + 10, height, int(height * 0.12)):
        draw.rectangle([width // 2 - 3, y, width // 2 + 3, y + int(height * 0.05)], fill=(240, 240, 240))
    draw.rectangle([int(width * 0.82), int(height * 0.35), int(width * 0.84), int(height * 0.7)], fill=(90, 60, 40))
    draw.ellipse([int(width * 0.74), int(height * 0.15), int(width * 0.92), int(height * 0.42)], fill=(40, 130, 60))
    return img
