# -*- coding: utf-8 -*-
"""
OpenAI-compatible backend for GeoAI-VLM
=======================================
Describe images with any server that speaks the OpenAI chat-completions
protocol: a vLLM server, Hugging Face TGI / Inference Endpoints, SGLang,
Ollama, LM Studio, or a hosted API.

Only ``requests`` is used. Requests run concurrently in a bounded thread pool,
with a timeout per request and retries with exponential backoff (honouring
``Retry-After``) for rate limits, server errors and dropped connections.

Credentials are read from an environment variable whose *name* you pass
(``api_key_env``). By default no key is sent at all, so a key set for one
service is never forwarded to another server by accident. The key is never
written to a log, a record, an exception message or a model input.
"""

from __future__ import annotations

import email.utils
import logging
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlsplit, urlunsplit

import requests

from .chat import GenerationOutput, build_chat_messages, image_to_data_url
from .describer import _ChatBackend, _merge_alias, _looks_like_system_role_error


__all__ = ["OpenAICompatibleBackend", "OpenAICompatibleError"]

logger = logging.getLogger(__name__)

#: Statuses worth retrying: timeouts, rate limits, transient server errors.
RETRY_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
#: Statuses that mean the configuration is wrong; every request would fail.
FATAL_STATUSES = frozenset({401, 403, 404})

_STRUCTURED_HINTS = ("response_format", "json_schema", "guided", "grammar", "structured")


class OpenAICompatibleError(RuntimeError):
    """An HTTP error from the server, with the key and credentials removed."""

    def __init__(self, status: Optional[int], message: str):
        super().__init__(message)
        self.status = status


def _strip_userinfo(url: str) -> str:
    parts = urlsplit(url)
    if parts.username or parts.password:
        host = parts.hostname or ""
        if parts.port:
            host = f"{host}:{parts.port}"
        parts = parts._replace(netloc=host)
    return urlunsplit(parts)


def _retry_after_seconds(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, when.timestamp() - time.time())


def _message_text(message: Dict[str, Any]) -> str:
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, list):  # some servers return content parts
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)


class OpenAICompatibleBackend(_ChatBackend):
    """Backend for OpenAI-compatible chat-completions servers.

    Args:
        model: Model name as the server knows it.
        base_url: API root including the version segment, e.g.
            ``http://localhost:8000/v1`` (vLLM server), ``http://localhost:11434/v1``
            (Ollama), ``http://localhost:1234/v1`` (LM Studio).
        api_key_env: *Name* of the environment variable holding the API key
            (e.g. ``"HF_TOKEN"``, ``"OPENAI_API_KEY"``). ``None`` (default)
            sends no key, which is right for most local servers.
        max_new_tokens: Maximum generated tokens (``max_tokens`` is the older
            name). Sent under ``max_tokens_field``.
        temperature, top_p, seed: Sampling settings sent to the server.
        timeout: Read timeout per request, in seconds.
        connect_timeout: Connection timeout, in seconds.
        max_retries: Retries after the first attempt for retryable failures.
        backoff_factor: Base delay; attempt *n* waits about
            ``backoff_factor * 2**n`` seconds (with jitter), capped at
            ``max_backoff``. ``Retry-After`` takes precedence.
        max_concurrency: Requests in flight at once.
        image_max_side: Downscale images so the longer side is at most this
            many pixels before upload (``None`` sends the file as is).
        max_tokens_field: ``"max_tokens"`` (default) or
            ``"max_completion_tokens"`` for servers that require it.
        system_prompt_mode: ``"auto"`` sends a system turn and, if the server
            rejects it as unsupported by the chat template, retries with the
            system text prepended. ``"system"`` / ``"prepend"`` force a form.
        structured_output: Send ``response_format`` with the JSON schema. If
            the server rejects it, falls back to unconstrained decoding.
        json_schema: The response schema.
        revision: Model revision the server runs, if you know it. It is
            recorded as given; the server's actual revision cannot be verified.
        extra_body: Extra fields merged into every request body.
        headers: Extra request headers (never logged).
        trust_remote_code: Has no effect -- the server decides what it runs.
    """

    name = "openai"
    _image_style = "openai"
    _system_content = "string"

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: str = "http://localhost:8000/v1",
        api_key_env: Optional[str] = None,
        *,
        model_name: Optional[str] = None,
        max_tokens: int = 2048,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        timeout: float = 120.0,
        connect_timeout: float = 10.0,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        max_backoff: float = 30.0,
        max_concurrency: int = 4,
        image_max_side: Optional[int] = None,
        max_tokens_field: str = "max_tokens",
        system_prompt_mode: str = "auto",
        structured_output: bool = False,
        json_schema: Optional[Dict[str, Any]] = None,
        revision: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        trust_remote_code: bool = False,
    ):
        model = model or model_name
        if not model:
            raise ValueError("OpenAICompatibleBackend needs a model name")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if max_tokens_field not in ("max_tokens", "max_completion_tokens"):
            raise ValueError("max_tokens_field must be 'max_tokens' or 'max_completion_tokens'")
        self._init_common(
            model,
            max_new_tokens=_merge_alias("max_new_tokens", max_new_tokens, "max_tokens", max_tokens, 2048),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            seed=seed,
            system_prompt_mode=system_prompt_mode,
            structured_output=structured_output,
            json_schema=json_schema,
            trust_remote_code=False,  # meaningless client-side
            revision=revision,
        )
        if trust_remote_code:
            logger.info("trust_remote_code has no effect on an OpenAI-compatible server")
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout = float(timeout)
        self.connect_timeout = float(connect_timeout)
        self.max_retries = int(max_retries)
        self.backoff_factor = float(backoff_factor)
        self.max_backoff = float(max_backoff)
        self.max_concurrency = int(max_concurrency)
        self.image_max_side = image_max_side
        self.max_tokens_field = max_tokens_field
        self.extra_body = dict(extra_body or {})
        self._headers = dict(headers or {})

        self._local = threading.local()
        self._lock = threading.Lock()
        self._structured_supported: Optional[bool] = None
        self._fingerprint: Optional[str] = None
        self._warned_missing_key = False

    @property
    def model(self) -> str:
        return self.model_name

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(model={self.model_name!r}, "
            f"base_url={_strip_userinfo(self.base_url)!r}, api_key_env={self.api_key_env!r})"
        )

    # -- credentials ------------------------------------------------------
    def _api_key(self) -> Optional[str]:
        if not self.api_key_env:
            return None
        key = os.environ.get(self.api_key_env)
        if not key and not self._warned_missing_key:
            logger.warning(
                "environment variable %s is not set; sending requests without a key",
                self.api_key_env,
            )
            self._warned_missing_key = True
        return key or None

    def _redact(self, text: str) -> str:
        secrets = [self._api_key_silent()] + [v for v in self._headers.values() if v]
        for secret in secrets:
            if secret and len(secret) >= 4:
                text = text.replace(secret, "***")
        return text

    def _api_key_silent(self) -> Optional[str]:
        return os.environ.get(self.api_key_env) if self.api_key_env else None

    # -- BaseBackend --------------------------------------------------------
    def is_available(self) -> bool:
        """The client side is always available; reachability is checked per call."""
        return True

    def load_model(self) -> None:
        """Nothing to load locally; the server holds the model."""
        return None

    def backend_version(self) -> Optional[str]:
        version = f"openai-compatible client (requests {requests.__version__})"
        if self._fingerprint:
            version += f"; server fingerprint {self._fingerprint}"
        return version

    @property
    def model_revision(self) -> Optional[str]:
        """The revision given by the caller; a server's revision is not observable."""
        return self.revision

    def generation_params(self) -> Dict[str, Any]:
        params = super().generation_params()
        params["image_max_side"] = self.image_max_side
        return params

    # -- HTTP -------------------------------------------------------------
    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            self._local.session = session
        return session

    def _sleep_before_retry(self, attempt: int, retry_after: Optional[float]) -> None:
        if retry_after is not None:
            delay = min(self.max_backoff, retry_after)
        else:
            delay = min(self.max_backoff, self.backoff_factor * (2 ** attempt))
            delay *= 0.5 + random.random() / 2  # jitter
        time.sleep(delay)

    def _request(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        safe_url = _strip_userinfo(url)
        headers = {"Content-Type": "application/json", **self._headers}
        key = self._api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"

        attempt = 0
        while True:
            try:
                response = self._session().request(
                    method,
                    url,
                    json=payload,
                    headers=headers,
                    timeout=(self.connect_timeout, self.timeout),
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                if attempt < self.max_retries:
                    logger.info(
                        "%s %s failed (%s); retry %d/%d",
                        method, safe_url, type(exc).__name__, attempt + 1, self.max_retries,
                    )
                    self._sleep_before_retry(attempt, None)
                    attempt += 1
                    continue
                raise OpenAICompatibleError(
                    None,
                    f"{method} {safe_url} failed after {attempt + 1} attempt(s): {type(exc).__name__}",
                ) from None

            status = response.status_code
            if status in RETRY_STATUSES and attempt < self.max_retries:
                retry_after = _retry_after_seconds(response.headers.get("Retry-After"))
                logger.info(
                    "%s %s returned HTTP %d; retry %d/%d",
                    method, safe_url, status, attempt + 1, self.max_retries,
                )
                self._sleep_before_retry(attempt, retry_after)
                attempt += 1
                continue
            if status >= 400:
                body = self._redact(response.text[:500])
                raise OpenAICompatibleError(status, f"HTTP {status} from {safe_url}: {body}")
            try:
                return response.json()
            except ValueError:
                raise OpenAICompatibleError(status, f"non-JSON response from {safe_url}") from None

    def list_models(self) -> List[str]:
        """Model ids the server reports (``GET /models``)."""
        data = self._request("GET", "models")
        return [m.get("id") for m in data.get("data", []) if isinstance(m, dict)]

    def complete(self, messages: List[Dict[str, Any]], **overrides: Any) -> str:
        """Send an arbitrary chat and return the reply text.

        This is the low-level call used for descriptions; it is public so an
        application can reuse the configured client (for example a text-only
        question answered from retrieved descriptions).
        """
        payload = self._payload(messages, structured=False)
        payload.update(overrides)
        data = self._request("POST", "chat/completions", payload)
        return self._reply_text(data)

    def _payload(self, messages: List[Dict[str, Any]], structured: bool) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            self.max_tokens_field: self.max_new_tokens,
            "temperature": self.temperature,
        }
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.seed is not None:
            payload["seed"] = self.seed
        if structured:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "geoai_vlm_response",
                    "schema": self.json_schema,
                    "strict": True,
                },
            }
        payload.update(self.extra_body)
        return payload

    def _reply_text(self, data: Dict[str, Any]) -> str:
        fingerprint = data.get("system_fingerprint")
        if fingerprint:
            self._fingerprint = str(fingerprint)
        choices = data.get("choices") or []
        if not choices:
            raise OpenAICompatibleError(None, "response contained no choices")
        return _message_text(choices[0].get("message") or {})

    # -- descriptions -------------------------------------------------------
    def _messages(self, image: Any, system_prompt: str, user_prompt: str, mode: str):
        encoded = image_to_data_url(image, max_side=self.image_max_side)
        return build_chat_messages(
            user_prompt,
            encoded,
            system_prompt,
            mode=mode,
            image_style="openai",
            system_content="string",
        )

    def _describe_one(self, image: Any, system_prompt: str, user_prompt: str) -> GenerationOutput:
        mode = self._system_mode_for(system_prompt)
        try:
            messages = self._messages(image, system_prompt, user_prompt, mode)
        except Exception as exc:
            return GenerationOutput(
                text="",
                error=f"image could not be read: {type(exc).__name__}: {exc}",
                decoding_mode="unconstrained",
                system_prompt_mode=mode,
            )

        while True:
            structured = self.wants_structured and self._structured_supported is not False
            try:
                data = self._request("POST", "chat/completions", self._payload(messages, structured))
                return GenerationOutput(
                    text=self._reply_text(data),
                    decoding_mode="json_schema" if structured else "unconstrained",
                    system_prompt_mode=mode,
                )
            except OpenAICompatibleError as exc:
                message = str(exc).lower()
                if exc.status in (400, 422) and structured and any(h in message for h in _STRUCTURED_HINTS):
                    with self._lock:
                        if self._structured_supported is not False:
                            logger.warning(
                                "server rejected response_format; continuing with "
                                "unconstrained decoding"
                            )
                        self._structured_supported = False
                    continue
                if (
                    exc.status in (400, 422)
                    and mode == "system"
                    and self.system_prompt_mode == "auto"
                    and _looks_like_system_role_error(exc)
                ):
                    with self._lock:
                        self._fall_back_to_prepend(
                            system_prompt, f"server rejected the system role (HTTP {exc.status})"
                        )
                    mode = "prepend"
                    messages = self._messages(image, system_prompt, user_prompt, mode)
                    continue
                raise

    def generate_outputs(
        self,
        images: Sequence[Any],
        system_prompt: str,
        user_prompt: str,
    ) -> List[GenerationOutput]:
        """Describe images concurrently; results keep the input order.

        A failure confined to one image (unreadable file, a 4xx about that
        request, retries exhausted) becomes that item's error. A configuration
        failure (401/403/404), or every item failing to reach the server,
        raises instead of filling a table with errors.
        """
        if not images:
            return []

        def run(image):
            try:
                return self._describe_one(image, system_prompt, user_prompt), None
            except OpenAICompatibleError as exc:
                return None, exc

        workers = min(self.max_concurrency, len(images))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(run, images))

        errors = [exc for _, exc in results if exc is not None]
        for exc in errors:
            if exc.status in FATAL_STATUSES:
                raise exc
        if errors and len(errors) == len(images) and all(e.status is None for e in errors):
            raise errors[0]

        mode_now = self._system_modes.get(system_prompt or "", ("system",))[0]
        outputs: List[GenerationOutput] = []
        for output, exc in results:
            if output is None:
                output = GenerationOutput(
                    text="",
                    error=str(exc),
                    decoding_mode="unconstrained",
                    system_prompt_mode=mode_now,
                )
            outputs.append(output)
        return outputs

    def generate(
        self,
        image_paths: List[str],
        system_prompt: str,
        user_prompt: str,
    ) -> List[str]:
        """Generate descriptions for a batch of images, in input order."""
        return [o.text for o in self.generate_outputs(image_paths, system_prompt, user_prompt)]
