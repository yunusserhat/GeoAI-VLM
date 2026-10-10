# -*- coding: utf-8 -*-
"""
GeoAI-VLM research demonstration (Gradio)
=========================================
A small local interface over :class:`geoai_vlm.service.DemoService`:

* **Describe an image** -- upload a street-level image; a VLM (local or an
  OpenAI-compatible endpoint) returns a structured description and the
  ``active_mobility_audit_v1`` observations; similar indexed scenes are
  listed and mapped with their recorded observations.
* **Scenes near a location** -- enter a coordinate; the nearest indexed scenes
  are listed and mapped.
* **Ask the indexed descriptions** -- questions are answered only from
  retrieved descriptions, every statement cites an image id, and the tool
  declines when the index holds no supporting record.

Run it locally::

    geoai-vlm app --index ./demo_index --model HuggingFaceTB/SmolVLM2-500M-Video-Instruct

The interface is a research demonstration: it shows model outputs and
retrieval, not assessments, and makes no design or health claims. It binds to
127.0.0.1 by default and never creates a public share link.

Requires ``pip install 'geoai-vlm[app]'``.
"""

from __future__ import annotations

import html
import json
from typing import Any, Optional, Tuple

import pandas as pd

from .service import DEMO_DISCLAIMER, DemoService


__all__ = ["build_app", "scene_map_html", "launch"]

_LEAFLET = "https://unpkg.com/leaflet@1.9.4/dist"
_SHOW_COLUMNS = ("image_id", "similarity", "distance_m", "lat", "lon", "scene_narrative")


def scene_map_html(records: pd.DataFrame, centre: Optional[Tuple[float, float]] = None, height: int = 360) -> str:
    """A Leaflet map (in an iframe) of records with ``lat``/``lon``.

    Popups show the image id and the record's observed audit states. All text
    is HTML-escaped; tiles are OpenStreetMap's, with attribution.
    """
    rows = records.dropna(subset=["lat", "lon"]) if len(records) else records
    markers = []
    for _, row in rows.iterrows():
        observed = [
            f"{c[len('audit_'):-len('_state')]}: {row[c]}"
            for c in row.index
            if c.startswith("audit_") and c.endswith("_state") and pd.notna(row[c])
        ]
        popup = f"<b>{html.escape(str(row.get('image_id', '')))}</b>"
        if observed:
            popup += "<br>" + "<br>".join(html.escape(o) for o in observed[:12])
        markers.append({"lat": float(row["lat"]), "lon": float(row["lon"]), "popup": popup})
    if centre is None and markers:
        centre = (sum(m["lat"] for m in markers) / len(markers), sum(m["lon"] for m in markers) / len(markers))
    centre = centre or (0.0, 0.0)
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="{_LEAFLET}/leaflet.css">
<script src="{_LEAFLET}/leaflet.js"></script>
<style>html,body,#m{{height:100%;margin:0}}</style></head><body><div id="m"></div><script>
var map = L.map('m').setView([{centre[0]:.6f}, {centre[1]:.6f}], {16 if markers else 2});
L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{maxZoom: 19,
  attribution: '&copy; OpenStreetMap contributors'}}).addTo(map);
var pts = {json.dumps(markers)};
pts.forEach(function(p) {{ L.marker([p.lat, p.lon]).addTo(map).bindPopup(p.popup); }});
if (pts.length > 1) {{ map.fitBounds(pts.map(function(p) {{ return [p.lat, p.lon]; }}), {{padding: [20, 20]}}); }}
</script></body></html>"""
    return (
        f'<iframe style="width:100%;height:{height}px;border:0" '
        f'srcdoc="{html.escape(page, quote=True)}"></iframe>'
    )


def _table(records: pd.DataFrame) -> pd.DataFrame:
    if records is None or len(records) == 0:
        return pd.DataFrame(columns=["image_id"])
    keep = [c for c in _SHOW_COLUMNS if c in records.columns]
    keep += [c for c in records.columns if c.startswith("audit_") and c.endswith("_state")][:8]
    return records[keep]


def build_app(service: DemoService, title: str = "GeoAI-VLM research demo"):
    """Build (but do not launch) the Gradio interface for *service*."""
    try:
        import gradio as gr
    except ImportError as exc:
        raise ImportError("the demo needs Gradio: pip install 'geoai-vlm[app]'") from exc

    def describe(image):
        if image is None:
            return {}, pd.DataFrame(), _table(None), scene_map_html(pd.DataFrame())
        if service.describer is None:
            raise gr.Error("No description model configured.")
        result = service.describe(image)
        similar = service.similar_scenes(image=image) if service.index is not None else pd.DataFrame()
        details = {"description": result["description"], "provenance": result["provenance"]}
        return details, result["observations"], _table(similar), scene_map_html(similar)

    def near(lat, lon, k):
        if service.index is None:
            raise gr.Error("No scene index configured.")
        try:
            records = service.scenes_near(float(lat), float(lon), int(k))
        except (TypeError, ValueError) as exc:
            raise gr.Error(str(exc))
        return _table(records), scene_map_html(records, centre=(float(lat), float(lon)))

    def ask(question, history):
        history = list(history or [])
        if service.index is None:
            raise gr.Error("No scene index configured.")
        result = service.answer(question)
        if result.refused:
            reply = f"No answer: {result.reason}."
        else:
            reply = result.answer + f"\n\n_Sources: {', '.join(result.citations)}_"
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": reply}]
        return history, "", _table(result.records)

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}\n\n> {DEMO_DISCLAIMER}")
        with gr.Tab("Describe an image"):
            with gr.Row():
                image = gr.Image(type="pil", label="Street-level image")
                with gr.Column():
                    run = gr.Button("Describe", variant="primary")
                    details = gr.JSON(label="Structured description and provenance")
            observations = gr.Dataframe(label="Observed features (image observations, not assessments)")
            similar_table = gr.Dataframe(label="Similar indexed scenes")
            similar_map = gr.HTML()
            run.click(describe, inputs=image, outputs=[details, observations, similar_table, similar_map])
        with gr.Tab("Scenes near a location"):
            with gr.Row():
                lat = gr.Number(label="Latitude")
                lon = gr.Number(label="Longitude")
                k = gr.Slider(1, 25, value=service.k, step=1, label="Scenes")
            go = gr.Button("Find scenes", variant="primary")
            near_table = gr.Dataframe(label="Nearest indexed scenes")
            near_map = gr.HTML()
            go.click(near, inputs=[lat, lon, k], outputs=[near_table, near_map])
        with gr.Tab("Ask the indexed descriptions"):
            gr.Markdown(
                "Answers use only retrieved descriptions and cite image ids. "
                "When no indexed record supports an answer, none is given."
            )
            chat = gr.Chatbot(label="Answers with sources", height=360)
            question = gr.Textbox(label="Question", placeholder="Where do images show cycle lanes?")
            sources = gr.Dataframe(label="Retrieved records")
            question.submit(ask, inputs=[question, chat], outputs=[chat, question, sources])
    return demo


def launch(service: DemoService, host: str = "127.0.0.1", port: int = 7860, **kwargs: Any):
    """Build and launch locally (no public share link)."""
    demo = build_app(service)
    kwargs.pop("share", None)
    return demo.launch(server_name=host, server_port=port, share=False, **kwargs)
