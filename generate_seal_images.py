#!/usr/bin/env python3
"""Generate sealified replacements for all images referenced in index.html using OpenAI image edits."""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

PROJECT_DIR = Path(__file__).resolve().parent
HTML_PATH = PROJECT_DIR / "index.html"
OUT_DIR = PROJECT_DIR / "sealified"
MANIFEST_PATH = OUT_DIR / "manifest.json"
MODEL = "gpt-image-1"
SIZE = "1536x1024"
QUALITY = "low"
SLEEP_SECONDS = 1.0


def build_entries() -> list[dict]:
    soup = BeautifulSoup(HTML_PATH.read_text(), "html.parser")
    entries = []
    seen = set()

    for idx, img in enumerate(soup.find_all("img"), start=1):
        src = img.get("src")
        if not src or src in seen or src.startswith("sealified/"):
            continue
        seen.add(src)

        alt = (img.get("alt") or "").strip()
        card = img.find_parent(class_=[
            "photo-card",
            "crab-act",
            "night-swim-shot",
            "bob-shot",
            "memory-beach-shot",
            "gratje-shot",
        ])
        heading = ""
        caption = ""
        label = ""
        if card:
            heading_el = card.find(["h3"])
            label_el = card.find(class_=["act-label", "shot-label", "bob-label", "beach-label", "gratje-label"])
            caption_el = card.find("p")
            heading = heading_el.get_text(" ", strip=True) if heading_el else ""
            label = label_el.get_text(" ", strip=True) if label_el else ""
            caption = caption_el.get_text(" ", strip=True) if caption_el else ""

        section = img.find_parent("section")
        section_title = ""
        if section:
            h2 = section.find("h2")
            section_title = h2.get_text(" ", strip=True) if h2 else ""

        mood_bits = [bit for bit in [section_title, heading, label, caption] if bit]
        mood = " | ".join(mood_bits)
        prompt = (
            "Edit this photo into a sealified remake for a friendship-trip website. "
            "Preserve the overall composition, camera angle, lighting mood, and tropical-travel energy, "
            "but replace visible humans or animals with expressive, photorealistic seals acting out the same moment. "
            f"Subject hint: {alt or src}. "
            f"Story hint: {mood or 'friends in Thailand making memories'}. "
            "Keep it funny, beautiful, coherent, and believable. No text, no watermark, no split-screen, no collage."
        )
        entries.append(
            {
                "src": src,
                "prompt": prompt,
                "alt": alt,
                "section": section_title,
                "context": mood,
            }
        )

    return entries


def generate_edit(image_path: Path, prompt: str, api_key: str) -> bytes:
    url = "https://api.openai.com/v1/images/edits"
    with image_path.open("rb") as f:
        files = {
            "image[]": (image_path.name, f.read(), "image/jpeg"),
        }
    data = {
        "model": MODEL,
        "prompt": prompt,
        "size": SIZE,
        "quality": QUALITY,
    }
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        files=files,
        data=data,
        timeout=600,
    )
    response.raise_for_status()
    payload = response.json()
    return base64.b64decode(payload["data"][0]["b64_json"])


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required")

    OUT_DIR.mkdir(exist_ok=True)
    entries = build_entries()
    manifest = []
    total = len(entries)

    for index, entry in enumerate(entries, start=1):
        source = PROJECT_DIR / entry["src"]
        destination = OUT_DIR / (Path(entry["src"]).stem + ".png")
        print(f"[{index}/{total}] {entry['src']} -> {destination.name}", flush=True)
        if destination.exists() and destination.stat().st_size > 0:
            print("  exists, skipping", flush=True)
        else:
            image_bytes = generate_edit(source, entry["prompt"], api_key)
            destination.write_bytes(image_bytes)
            print(f"  wrote {destination}", flush=True)
            time.sleep(SLEEP_SECONDS)
        manifest.append({**entry, "output": str(destination.relative_to(PROJECT_DIR))})

    MANIFEST_PATH.write_text(json.dumps({"images": manifest}, indent=2))
    print(f"Wrote manifest: {MANIFEST_PATH}", flush=True)


if __name__ == "__main__":
    main()
