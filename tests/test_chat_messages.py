# -*- coding: utf-8 -*-
"""
Model-agnostic chat construction (P0-A item 1).

Chat templates are simulated with plain functions so the tests need neither
jinja2 nor transformers: one template accepts a system turn, one raises on it,
one silently drops it. No model, network or API key is used.
"""

from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from geoai_vlm.chat import (
    build_chat_messages,
    image_to_data_url,
    load_image,
    make_synthetic_street_image,
    resolve_system_prompt_mode,
)


def _text_of(message):
    content = message["content"]
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if p.get("type") == "text")


# ---------------------------------------------------------------------------
# Simulated chat templates
# ---------------------------------------------------------------------------
def template_with_system(messages):
    return "".join(f"<{m['role']}>{_text_of(m)}" for m in messages) + "<assistant>"


def template_rejecting_system(messages):
    if any(m["role"] == "system" for m in messages):
        raise ValueError("System role not supported")
    return template_with_system(messages)


def template_dropping_system(messages):
    return "".join(
        f"<{m['role']}>{_text_of(m)}" for m in messages if m["role"] != "system"
    )


# ---------------------------------------------------------------------------
# build_chat_messages
# ---------------------------------------------------------------------------
class TestBuildChatMessages:
    def test_system_mode_sends_a_system_turn(self):
        msgs = build_chat_messages("Describe.", None, "Be precise.", mode="system")
        assert [m["role"] for m in msgs] == ["system", "user"]
        assert msgs[0]["content"] == [{"type": "text", "text": "Be precise."}]
        assert _text_of(msgs[1]) == "Describe."

    def test_prepend_mode_moves_system_text_into_the_user_turn(self):
        msgs = build_chat_messages("Describe.", None, "Be precise.", mode="prepend")
        assert [m["role"] for m in msgs] == ["user"]
        assert _text_of(msgs[0]) == "Be precise.\n\nDescribe."

    def test_no_system_prompt_gives_user_turn_only(self):
        for mode in ("system", "prepend", "auto"):
            msgs = build_chat_messages("Describe.", None, None, mode=mode)
            assert [m["role"] for m in msgs] == ["user"]
            assert _text_of(msgs[0]) == "Describe."

    def test_auto_is_treated_as_try_system_first(self):
        msgs = build_chat_messages("u", None, "s", mode="auto")
        assert msgs[0]["role"] == "system"

    def test_system_content_as_string(self):
        msgs = build_chat_messages("u", None, "s", system_content="string")
        assert msgs[0] == {"role": "system", "content": "s"}

    def test_image_precedes_text_in_the_user_turn(self):
        img = make_synthetic_street_image(32, 24)
        msgs = build_chat_messages("u", img, "s", image_style="hf")
        parts = msgs[-1]["content"]
        assert parts[0]["type"] == "image"
        assert isinstance(parts[0]["image"], Image.Image)
        assert parts[1] == {"type": "text", "text": "u"}

    def test_openai_style_attaches_a_data_url(self):
        img = make_synthetic_street_image(32, 24)
        part = build_chat_messages("u", img, image_style="openai")[-1]["content"][0]
        assert part["type"] == "image_url"
        assert part["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_vllm_style_attaches_pil(self):
        img = make_synthetic_street_image(32, 24)
        part = build_chat_messages("u", img, image_style="vllm_pil")[-1]["content"][0]
        assert part["type"] == "image_pil"
        assert isinstance(part["image_pil"], Image.Image)

    def test_remote_url_is_passed_through(self):
        url = "https://example.org/a.jpg"
        assert build_chat_messages("u", url, image_style="hf")[-1]["content"][0] == {
            "type": "image",
            "url": url,
        }
        assert build_chat_messages("u", url, image_style="openai")[-1]["content"][0][
            "image_url"
        ]["url"] == url

    @pytest.mark.parametrize("bad", ["sys", "", "SYSTEM"])
    def test_invalid_mode_is_rejected(self, bad):
        with pytest.raises(ValueError):
            build_chat_messages("u", None, "s", mode=bad)

    def test_invalid_image_style_is_rejected(self):
        with pytest.raises(ValueError):
            build_chat_messages("u", make_synthetic_street_image(8, 8), image_style="other")


# ---------------------------------------------------------------------------
# resolve_system_prompt_mode
# ---------------------------------------------------------------------------
class TestResolveSystemPromptMode:
    def test_template_with_system_role_keeps_it(self):
        mode, reason = resolve_system_prompt_mode("auto", "Be precise.", template_with_system)
        assert mode == "system"
        assert reason is None

    def test_template_rejecting_system_role_falls_back_to_prepend(self):
        mode, reason = resolve_system_prompt_mode("auto", "Be precise.", template_rejecting_system)
        assert mode == "prepend"
        assert "rejected" in reason and "System role not supported" in reason

    def test_template_silently_dropping_system_role_falls_back_to_prepend(self):
        mode, reason = resolve_system_prompt_mode("auto", "Be precise.", template_dropping_system)
        assert mode == "prepend"
        assert "dropped" in reason

    def test_fallback_is_logged(self, caplog):
        with caplog.at_level("WARNING", logger="geoai_vlm.chat"):
            resolve_system_prompt_mode("auto", "s", template_rejecting_system)
        assert any("prepended" in r.getMessage() for r in caplog.records)

    @pytest.mark.parametrize("requested", ["system", "prepend"])
    def test_explicit_modes_are_not_probed(self, requested):
        def explode(_):
            raise AssertionError("explicit modes must not render the template")

        assert resolve_system_prompt_mode(requested, "s", explode) == (requested, None)

    @pytest.mark.parametrize("empty", [None, "", "   "])
    def test_no_system_prompt_is_none(self, empty):
        assert resolve_system_prompt_mode("auto", empty, template_with_system) == ("none", None)

    def test_auto_without_renderer_tries_system_and_says_so(self):
        mode, reason = resolve_system_prompt_mode("auto", "s", None)
        assert mode == "system"
        assert "not inspectable" in reason

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            resolve_system_prompt_mode("whatever", "s", None)

    def test_prepend_fallback_messages_reach_the_template(self):
        """End to end: after the fallback the rejecting template renders fine."""
        mode, _ = resolve_system_prompt_mode("auto", "Be precise.", template_rejecting_system)
        msgs = build_chat_messages("Describe.", None, "Be precise.", mode=mode)
        rendered = template_rejecting_system(msgs)
        assert "Be precise." in rendered and "Describe." in rendered


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
class TestImages:
    def test_jpeg_file_is_sent_byte_for_byte(self, tmp_path):
        path = tmp_path / "a.jpg"
        make_synthetic_street_image(40, 30).save(path, quality=80)
        url = image_to_data_url(path)
        assert url.startswith("data:image/jpeg;base64,")
        assert base64.b64decode(url.split(",", 1)[1]) == path.read_bytes()

    def test_max_side_downscales_preserving_aspect(self, tmp_path):
        path = tmp_path / "b.png"
        make_synthetic_street_image(400, 200).save(path)
        url = image_to_data_url(path, max_side=100)
        decoded = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
        assert decoded.size == (100, 50)

    def test_load_image_accepts_bytes_pil_and_data_url(self, tmp_path):
        img = make_synthetic_street_image(20, 10)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        assert load_image(buf.getvalue()).size == (20, 10)
        assert load_image(img).size == (20, 10)
        data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        assert load_image(data_url).size == (20, 10)

    def test_load_image_converts_to_rgb(self):
        rgba = Image.new("RGBA", (4, 4))
        assert load_image(rgba).mode == "RGB"

    def test_load_image_does_not_download(self):
        with pytest.raises(ValueError, match="does not download"):
            load_image("https://example.org/a.jpg")

    def test_synthetic_image_is_deterministic(self):
        a = make_synthetic_street_image(64, 48).tobytes()
        b = make_synthetic_street_image(64, 48).tobytes()
        assert a == b
