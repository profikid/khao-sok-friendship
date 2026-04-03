#!/usr/bin/env python3
"""Generate sealified replacements using Replicate's Instant-ID for face-preserving seal transformations."""

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

# Replicate API settings
REPLICATE_API_URL = "https://api.replicate.com/v1/predictions"
INSTANT_ID_VERSION = "6af8583c541261472e92155d87bba80d5ad98461665802f2ba196ac099aaedc9"

# Seal oracle prompt template
SEAL_PROMPT = "a mystical seal oracle portrait, ancient hyper-intelligent seal creature with glowing eyes, ethereal underwater lighting, bioluminescent coral reef background, magical atmosphere, digital art style, highly detailed, 8k quality"


def build_entries() -> list[dict]:
    """Parse HTML and build list of images to sealify."""
    soup = BeautifulSoup(HTML_PATH.read_text(), "html.parser")
    entries = []
    seen = set()

    for img in soup.find_all("img"):
        src = img.get("src")
        if not src or src in seen or src.startswith("sealified/"):
            continue
        seen.add(src)

        alt = (img.get("alt") or "").strip()
        
        # Find parent context
        card = img.find_parent(class_=[
            "photo-card", "crab-act", "night-swim-shot", 
            "bob-shot", "memory-beach-shot", "gratje-shot"
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

        # Build contextual prompt
        context_parts = [p for p in [section_title, heading, label, caption, alt] if p]
        context = " | ".join(context_parts) if context_parts else "friends in Thailand making memories"
        
        # Customize seal prompt based on context
        custom_prompt = SEAL_PROMPT
        if "crab" in context.lower():
            custom_prompt += ", seal interacting with a hermit crab"
        elif "night" in context.lower() or "swim" in context.lower():
            custom_prompt += ", seal swimming at night under moonlight"
        elif "beach" in context.lower():
            custom_prompt += ", seal relaxing on tropical beach"
        elif "dog" in context.lower() or "gratje" in context.lower():
            custom_prompt += ", seal with a friendly dog companion"
        elif "beer" in context.lower() or "chang" in context.lower():
            custom_prompt += ", seal enjoying a tropical drink"
        elif "boat" in context.lower():
            custom_prompt += ", seal captain on a longtail boat"
        elif "bob" in context.lower() or "phone" in context.lower():
            custom_prompt += ", seal on a video call"

        entries.append({
            "src": src,
            "prompt": custom_prompt,
            "alt": alt,
            "section": section_title,
            "context": context,
        })

    return entries


def generate_sealified(image_path: Path, prompt: str, api_token: str, max_retries: int = 3) -> bytes:
    """Generate sealified image using Replicate Instant-ID via HTTP API."""
    
    # Read and encode image
    with image_path.open("rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    
    headers = {
        "Authorization": f"Token {api_token}",
        "Content-Type": "application/json",
    }
    
    # Create prediction
    payload = {
        "version": INSTANT_ID_VERSION,
        "input": {
            "image": f"data:image/jpeg;base64,{image_data}",
            "prompt": prompt,
            "negative_prompt": "blurry, low quality, distorted, ugly, bad anatomy, watermark, text, signature",
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "strength": 0.7,
        }
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(REPLICATE_API_URL, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 429:
                wait_time = 60 * (attempt + 1)
                print(f"\n    Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            
            response.raise_for_status()
            prediction = response.json()
            break
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 10 * (attempt + 1)
                print(f"\n    Error: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
    else:
        raise RuntimeError("Max retries exceeded")
    
    # Poll for completion
    prediction_id = prediction["id"]
    get_url = f"{REPLICATE_API_URL}/{prediction_id}"
    
    print(f"    Waiting...", end="", flush=True)
    
    while True:
        time.sleep(3)
        status_resp = requests.get(get_url, headers=headers, timeout=30)
        
        if status_resp.status_code == 429:
            print("R", end="", flush=True)
            time.sleep(30)
            continue
            
        status_resp.raise_for_status()
        status_data = status_resp.json()
        
        status = status_data.get("status")
        if status == "succeeded":
            print(" done!")
            output_url = status_data.get("output")
            if isinstance(output_url, list):
                output_url = output_url[0]
            
            # Download the result
            img_resp = requests.get(output_url, timeout=120)
            img_resp.raise_for_status()
            return img_resp.content
        elif status == "failed":
            error = status_data.get("error", "Unknown error")
            raise RuntimeError(f"Generation failed: {error}")
        elif status == "canceled":
            raise RuntimeError("Generation was canceled")
        else:
            print(".", end="", flush=True)


def main() -> None:
    """Main entry point."""
    api_token = os.getenv("REPLICATE_API_TOKEN") or os.getenv("REPLICATE_TOKEN")
    if not api_token:
        raise SystemExit("REPLICATE_API_TOKEN or REPLICATE_TOKEN is required. Set it as an environment variable.")

    OUT_DIR.mkdir(exist_ok=True)
    entries = build_entries()
    manifest = []
    total = len(entries)

    print(f"Found {total} images to process")
    print("=" * 60)

    for index, entry in enumerate(entries, start=1):
        source = PROJECT_DIR / entry["src"]
        destination = OUT_DIR / (Path(entry["src"]).stem + ".png")
        
        print(f"\n[{index}/{total}] {entry['src']}")
        
        if destination.exists() and destination.stat().st_size > 0:
            print(f"  ✓ Already exists")
        else:
            if not source.exists():
                print(f"  ✗ Source not found: {source}")
                continue
                
            try:
                image_bytes = generate_sealified(source, entry["prompt"], api_token)
                destination.write_bytes(image_bytes)
                print(f"  ✓ Saved ({len(image_bytes)} bytes)")
                time.sleep(5.0)  # Rate limiting between requests
            except Exception as e:
                print(f"  ✗ Error: {e}")
                if "rate" in str(e).lower() or "429" in str(e):
                    print("    Stopping due to rate limit. Run again later.")
                    break
                continue
        
        manifest.append({**entry, "output": str(destination.relative_to(PROJECT_DIR))})

    MANIFEST_PATH.write_text(json.dumps({"images": manifest, "total": len(manifest)}, indent=2))
    print(f"\n{'=' * 60}")
    print(f"Done! Wrote manifest: {MANIFEST_PATH}")
    print(f"Total images processed: {len(manifest)}")


if __name__ == "__main__":
    main()
