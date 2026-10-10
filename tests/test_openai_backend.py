# -*- coding: utf-8 -*-
"""
OpenAICompatibleBackend against a local fake HTTP server (P0-A item 4).

The server runs in a thread on 127.0.0.1 and speaks just enough of the
chat-completions protocol. Covered: success, 429 with retry, timeouts, fatal
configuration errors, structured-output fallback, system-role fallback,
ordering under concurrency, and that the API key never leaks into logs,
exceptions, records or model inputs. No external network is used.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pandas as pd
import pytest
from PIL import Image

from geoai_vlm.describer import ImageDescriber
from geoai_vlm.openai_compat import OpenAICompatibleBackend, OpenAICompatibleError

SECRET = "sk-test-SECRET-value-0123456789"


def _img(identity, size=(16, 16)):
    return Image.new("RGB", size, color=(identity, identity, identity))


def _identity_from_request(body):
    """The grey level of the image in the request (the fake's 'answer')."""
    for message in body["messages"]:
        if isinstance(message["content"], list):
            for part in message["content"]:
                if part.get("type") == "image_url":
                    data = part["image_url"]["url"].split(",", 1)[1]
                    with Image.open(io.BytesIO(base64.b64decode(data))) as img:
                        return img.convert("RGB").getpixel((0, 0))[0]
    return None


class FakeServer:
    """A configurable chat-completions server."""

    def __init__(self):
        self.requests = []
        self.lock = threading.Lock()
        self.script = []  # per-request overrides: (status, body, headers, delay)
        self.reject_system = False
        self.reject_response_format = False
        self.echo_auth_on_error = False
        self.delay_by_identity = {}

        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep test output clean
                pass

            def _reply(self, status, payload, headers=None):
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self):
                if self.path.endswith("/models"):
                    self._reply(200, {"data": [{"id": "fake-vlm"}]})
                else:
                    self._reply(404, {"error": "not found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                with server.lock:
                    server.requests.append({"headers": dict(self.headers), "body": body})
                    step = server.script.pop(0) if server.script else None
                if step is not None:
                    status, payload, headers, delay = step
                    if delay:
                        time.sleep(delay)
                    if status is not None:
                        if server.echo_auth_on_error:
                            payload = {"error": f"bad request, auth={self.headers.get('Authorization')}"}
                        return self._reply(status, payload, headers)
                if server.reject_response_format and "response_format" in body:
                    return self._reply(400, {"error": {"message": "response_format json_schema is not supported"}})
                if server.reject_system and any(m["role"] == "system" for m in body["messages"]):
                    return self._reply(400, {"message": "System role not supported", "type": "BadRequestError"})
                identity = _identity_from_request(body)
                delay = server.delay_by_identity.get(identity, 0)
                if delay:
                    time.sleep(delay)
                content = json.dumps({"description": f"image {identity}", "tags": ["fake"]})
                self._reply(
                    200,
                    {
                        "id": "x",
                        "model": body.get("model"),
                        "system_fingerprint": "fp-test",
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
                    },
                )

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base_url(self):
        host, port = self.httpd.server_address
        return f"http://{host}:{port}/v1"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    with FakeServer() as srv:
        yield srv


def _backend(server, **kwargs):
    kwargs.setdefault("backoff_factor", 0.01)
    kwargs.setdefault("max_backoff", 0.05)
    return OpenAICompatibleBackend("fake-vlm", base_url=server.base_url, **kwargs)


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------
class TestSuccess:
    def test_describes_an_image(self, server):
        out = _backend(server).generate_outputs([_img(5)], "sys", "describe")
        assert json.loads(out[0].text)["description"] == "image 5"
        assert out[0].error is None
        body = server.requests[0]["body"]
        assert body["model"] == "fake-vlm"
        assert body["max_tokens"] == 2048 and body["temperature"] == 0.0
        assert body["messages"][0] == {"role": "system", "content": "sys"}
        image_part = body["messages"][1]["content"][0]
        assert image_part["type"] == "image_url"
        assert image_part["image_url"]["url"].startswith("data:image/")

    def test_order_is_preserved_under_concurrency(self, server):
        server.delay_by_identity = {1: 0.3, 2: 0.0, 3: 0.15}
        backend = _backend(server, max_concurrency=3)
        out = backend.generate_outputs([_img(1), _img(2), _img(3)], "s", "u")
        assert [json.loads(o.text)["description"] for o in out] == ["image 1", "image 2", "image 3"]

    def test_concurrency_is_bounded(self, server):
        server.delay_by_identity = {i: 0.2 for i in range(1, 5)}
        backend = _backend(server, max_concurrency=2)
        start = time.monotonic()
        backend.generate_outputs([_img(i) for i in range(1, 5)], "s", "u")
        # Four 0.2 s requests, two at a time: at least two rounds.
        assert time.monotonic() - start >= 0.38

    def test_no_key_is_sent_by_default(self, server, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", SECRET)
        _backend(server).generate_outputs([_img(1)], "s", "u")
        assert "Authorization" not in server.requests[0]["headers"]

    def test_key_is_read_from_the_named_variable(self, server, monkeypatch):
        monkeypatch.setenv("MY_ENDPOINT_KEY", SECRET)
        _backend(server, api_key_env="MY_ENDPOINT_KEY").generate_outputs([_img(1)], "s", "u")
        assert server.requests[0]["headers"]["Authorization"] == f"Bearer {SECRET}"

    def test_image_downscaling_before_upload(self, server):
        _backend(server, image_max_side=20).generate_outputs([_img(9, (200, 100))], "s", "u")
        part = server.requests[0]["body"]["messages"][1]["content"][0]
        data = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
        assert Image.open(io.BytesIO(data)).size == (20, 10)

    def test_list_models(self, server):
        assert _backend(server).list_models() == ["fake-vlm"]

    def test_complete_for_text_only_chats(self, server):
        backend = _backend(server)
        text = backend.complete([{"role": "user", "content": "hello"}])
        assert json.loads(text)["description"] == "image None"

    def test_fingerprint_is_recorded_in_backend_version(self, server):
        backend = _backend(server)
        backend.generate_outputs([_img(1)], "s", "u")
        assert "fp-test" in backend.backend_version()


# ---------------------------------------------------------------------------
# Retries, timeouts, fatal errors
# ---------------------------------------------------------------------------
class TestRetries:
    def test_429_is_retried_then_succeeds(self, server):
        server.script = [
            (429, {"error": "rate limited"}, {"Retry-After": "0"}, 0),
            (429, {"error": "rate limited"}, {}, 0),
        ]
        out = _backend(server, max_retries=3).generate_outputs([_img(2)], "s", "u")
        assert json.loads(out[0].text)["description"] == "image 2"
        assert len(server.requests) == 3

    def test_retries_are_bounded(self, server):
        server.script = [(503, {"error": "busy"}, {}, 0)] * 6
        backend = _backend(server, max_retries=2, max_concurrency=1)
        out = backend.generate_outputs([_img(2), _img(3)], "s", "u")
        # Two items x (1 attempt + 2 retries), then each item records its error.
        assert len(server.requests) == 6
        assert all(o.error and "HTTP 503" in o.error for o in out)
        assert all(o.text == "" for o in out)

    def test_timeout_is_retried_and_then_reported(self, server):
        server.script = [(None, None, None, 1.0)] * 4
        backend = _backend(server, timeout=0.2, max_retries=1)
        start = time.monotonic()
        with pytest.raises(OpenAICompatibleError, match="ReadTimeout|Timeout"):
            backend.generate_outputs([_img(1)], "s", "u")
        assert time.monotonic() - start < 5

    def test_timeout_then_success(self, server):
        server.script = [(None, None, None, 1.0)]
        backend = _backend(server, timeout=0.3, max_retries=2)
        out = backend.generate_outputs([_img(4)], "s", "u")
        assert json.loads(out[0].text)["description"] == "image 4"

    @pytest.mark.parametrize("status", [401, 403, 404])
    def test_configuration_errors_raise(self, server, status):
        server.script = [(status, {"error": "nope"}, {}, 0)] * 3
        with pytest.raises(OpenAICompatibleError) as exc:
            _backend(server).generate_outputs([_img(1)], "s", "u")
        assert exc.value.status == status

    def test_unreachable_server_raises(self):
        backend = OpenAICompatibleBackend(
            "m", base_url="http://127.0.0.1:9/v1", max_retries=0, connect_timeout=0.5
        )
        with pytest.raises(OpenAICompatibleError, match="failed after 1 attempt"):
            backend.generate_outputs([_img(1)], "s", "u")

    def test_a_bad_request_for_one_item_does_not_sink_the_batch(self, server):
        server.script = [(400, {"error": "image too large"}, {}, 0)]
        out = _backend(server, max_concurrency=1).generate_outputs([_img(1), _img(2)], "s", "u")
        assert out[0].error and "HTTP 400" in out[0].error
        assert json.loads(out[1].text)["description"] == "image 2"


# ---------------------------------------------------------------------------
# Fallbacks
# ---------------------------------------------------------------------------
class TestFallbacks:
    SCHEMA = {"type": "object", "properties": {"description": {"type": "string"}}}

    def test_structured_output_sends_response_format(self, server):
        backend = _backend(server, structured_output=True, json_schema=self.SCHEMA)
        out = backend.generate_outputs([_img(1)], "s", "u")
        fmt = server.requests[0]["body"]["response_format"]
        assert fmt["type"] == "json_schema"
        assert fmt["json_schema"]["schema"] == self.SCHEMA
        assert out[0].decoding_mode == "json_schema"

    def test_rejected_response_format_falls_back(self, server):
        server.reject_response_format = True
        backend = _backend(server, structured_output=True, json_schema=self.SCHEMA)
        out = backend.generate_outputs([_img(1)], "s", "u")
        assert out[0].decoding_mode == "unconstrained"
        assert out[0].error is None
        assert "response_format" not in server.requests[-1]["body"]
        # Remembered: the next call does not try again.
        backend.generate_outputs([_img(2)], "s", "u")
        assert "response_format" not in server.requests[-1]["body"]

    def test_rejected_system_role_falls_back_to_prepend(self, server):
        server.reject_system = True
        backend = _backend(server)
        out = backend.generate_outputs([_img(1)], "Be precise.", "Describe.")
        assert out[0].system_prompt_mode == "prepend"
        last = server.requests[-1]["body"]["messages"]
        assert [m["role"] for m in last] == ["user"]
        assert last[0]["content"][1]["text"] == "Be precise.\n\nDescribe."

    def test_explicit_system_mode_reports_the_error(self, server):
        server.reject_system = True
        out = _backend(server, system_prompt_mode="system").generate_outputs([_img(1)], "s", "u")
        assert out[0].error and "System role not supported" in out[0].error


# ---------------------------------------------------------------------------
# The key must not leak
# ---------------------------------------------------------------------------
class TestKeyHygiene:
    def test_key_never_reaches_logs_errors_records_or_inputs(self, server, monkeypatch, caplog, tmp_path):
        monkeypatch.setenv("MY_ENDPOINT_KEY", SECRET)
        server.echo_auth_on_error = True  # a hostile/buggy server echoing the header
        server.script = [
            (500, {}, {}, 0),
            (500, {}, {}, 0),
            (400, {}, {}, 0),
        ]
        path = tmp_path / "img.png"
        _img(3).save(path)

        with caplog.at_level(logging.DEBUG):
            d = ImageDescriber(
                model_name="fake-vlm",
                backend="openai",
                prompt_template="simple",
                base_url=server.base_url,
                api_key_env="MY_ENDPOINT_KEY",
                max_retries=1,
                backoff_factor=0.01,
            )
            df = d.describe(image_paths=[path], output_path=tmp_path / "out.parquet")

        assert SECRET not in caplog.text
        assert SECRET not in repr(d.backend)
        stored = pd.read_parquet(tmp_path / "out.parquet")
        for column in stored.columns:
            assert not stored[column].astype(str).str.contains(SECRET, regex=False).any(), column
        assert df.iloc[0]["generation_error"] and "***" in df.iloc[0]["generation_error"]
        for request in server.requests:
            assert SECRET not in json.dumps(request["body"])

    def test_error_messages_are_redacted(self, server, monkeypatch):
        monkeypatch.setenv("MY_ENDPOINT_KEY", SECRET)
        server.echo_auth_on_error = True
        server.script = [(401, {}, {}, 0)]
        with pytest.raises(OpenAICompatibleError) as exc:
            _backend(server, api_key_env="MY_ENDPOINT_KEY").generate_outputs([_img(1)], "s", "u")
        assert SECRET not in str(exc.value)
        assert "***" in str(exc.value)

    def test_credentials_in_the_url_are_not_echoed(self):
        backend = OpenAICompatibleBackend(
            "m", base_url="http://user:pa55word@127.0.0.1:9/v1", max_retries=0, connect_timeout=0.3
        )
        assert "pa55word" not in repr(backend)
        with pytest.raises(OpenAICompatibleError) as exc:
            backend.generate_outputs([_img(1)], "s", "u")
        assert "pa55word" not in str(exc.value)


# ---------------------------------------------------------------------------
# End to end through ImageDescriber
# ---------------------------------------------------------------------------
class TestDescriberIntegration:
    def test_records_carry_backend_provenance(self, server, tmp_path):
        paths = []
        for i in (1, 2):
            p = tmp_path / f"{i}.png"
            _img(i).save(p)
            paths.append(p)
        d = ImageDescriber(
            model_name="fake-vlm",
            backend="openai",
            prompt_template="simple",
            base_url=server.base_url,
            structured_output=True,
        )
        df = d.describe(image_paths=paths)
        assert list(df["scene_narrative"]) == ["image 1", "image 2"]
        assert set(df["backend"]) == {"openai"}
        assert set(df["decoding_mode"]) == {"json_schema"}
        assert set(df["system_prompt_mode_effective"]) == {"system"}
        assert df["model_revision"].isna().all(), "a server's revision is not observable"
        params = json.loads(df.iloc[0]["generation_params"])
        assert params["structured_output"] is True
        assert "json_schema_digest" in params

    def test_backend_name_aliases(self, server):
        for alias in ("openai", "openai_compatible", "openai-compatible"):
            d = ImageDescriber(model_name="m", backend=alias, base_url=server.base_url)
            assert isinstance(d.backend, OpenAICompatibleBackend)
