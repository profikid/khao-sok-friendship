#!/usr/bin/env python3
"""Generate fallback sealified images for photos without detectable faces using FLUX or similar."""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

PROJECT_DIR = Path(__file__).resolve().parent
OUT_DIR = PROJECT_DIR / "sealified"

# Use FLUX for images without faces
FLUX_VERSION = "black-forest-labs/flux-schnell"
REPLICATE_API_URL = "https://api.replicate.com/v1/predictions"

def get_missing_images() -> list[str]:
    """Find images that don't have sealified versions yet."""
    # Get all jpg sources from HTML
    soup = BeautifulSoup((PROJECT_DIR / "index.html").read_text(), "html.parser")
    all_sources = set()
    for img in soup.find_all("img"):
        src = img.get("src")
        if src and not src.startswith("sealified/"):
            all_sources.add(src)
    
    # Find which ones are missing
    missing = []
    for src in sorted(all_sources):
        seal_name = Path(src).stem + ".png"
        seal_path = OUT_DIR / seal_name
        if not seal_path.exists():
            missing.append(src)
    
    return missing


def generate_with_flux(prompt: str, api_token: str) -> bytes:
    """Generate image using FLUX for images without faces."""
    
    headers = {
        "Authorization": f"Token {api_token}",
        "Content-Type": "application/json",
    }
    
    # Get model version
    model_resp = requests.get(
        f"https://api.replicate.com/v1/models/{FLUX_VERSION}",
        headers=headers,
        timeout=30
    )
    model_resp.raise_for_status()
    latest_version = model_resp.json()["latest_version"]["id"]
    
    payload = {
        "version": latest_version,
        "input": {
            "prompt": prompt,
            "aspect_ratio": "16:9",
            "num_outputs": 1,
        }
    }
    
    response = requests.post(REPLICATE_API_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    prediction = response.json()
    
    # Poll for completion
    prediction_id = prediction["id"]
    get_url = f"{REPLICATE_API_URL}/{prediction_id}"
    
    print(f"    Waiting...", end="", flush=True)
    
    while True:
        time.sleep(2)
        status_resp = requests.get(get_url, headers=headers, timeout=30)
        status_resp.raise_for_status()
        status_data = status_resp.json()
        
        status = status_data.get("status")
        if status == "succeeded":
            print(" done!")
            output_url = status_data.get("output")
            if isinstance(output_url, list):
                output_url = output_url[0]
            
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
    """Generate fallback seal images."""
    api_token = os.getenv("REPLICATE_API_TOKEN") or os.getenv("REPLICATE_TOKEN")
    if not api_token:
        raise SystemExit("REPLICATE_API_TOKEN or REPLICATE_TOKEN is required.")

    missing = get_missing_images()
    print(f"Found {len(missing)} images needing fallback sealification")
    print("=" * 60)
    
    # Context-aware prompts for missing images
    prompts = {
        "img8.jpg": "A mystical seal oracle floating in dramatic sunset clouds over mountains, golden hour lighting, ethereal atmosphere, bioluminescent coral reef elements, digital art, highly detailed",
        "img14.jpg": "A seal enjoying fresh coconuts and tropical drinks on a beach, orchid flowers, turquoise water, Thailand paradise vibes, magical atmosphere, digital art",
        "memory-beach-1.jpg": "A mystical seal oracle overlooking a tropical beach horizon, turquoise ocean, limestone karsts in distance, ethereal underwater lighting, magical atmosphere, digital art",
        "memory-beach-2.jpg": "A seal relaxing on a pristine white sand beach, palm trees, tropical vibes, bioluminescent elements, magical atmosphere, digital art",
        "memory-beach-4.jpg": "A seal enjoying a beach bar atmosphere, tropical drinks, sunset colors, magical bioluminescent lighting, digital art style",
        "memory-beach-5.jpg": "A seal silhouette during golden hour sunset at the beach, dramatic orange and pink sky, ethereal magical atmosphere, digital art",
        "gratje5.jpg": "A mystical seal oracle walking alongside a friendly dog on a tropical beach, palm trees, ocean waves, magical bioluminescent lighting, digital art",
        "gratje6.jpg": "A seal relaxing under palm trees on a tropical beach, ocean breeze, magical atmosphere with glowing coral elements, digital art",
    }
    
    for src in missing:
        destination = OUT_DIR / (Path(src).stem + ".png")
        print(f"\nGenerating fallback for: {src}")
        
        prompt = prompts.get(src, "A mystical seal oracle portrait, ancient hyper-intelligent seal creature, ethereal underwater lighting, bioluminescent coral reef background, magical atmosphere, digital art style")
        
        try:
            image_bytes = generate_with_flux(prompt, api_token)
            destination.write_bytes(image_bytes)
            print(f"  ✓ Saved ({len(image_bytes)} bytes)")
            time.sleep(3.0)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    print(f"\n{'=' * 60}")
    print("Fallback generation complete!")


if __name__ == "__main__":
    main()
