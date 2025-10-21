#!/usr/bin/env python3
"""
gemini_direct_image_description.py

Make direct API calls to Gemini to extract descriptions from images.
Uses the same model, prompt, and image processing as the batch pipeline.

Usage:
    # Developer API (requires GEMINI_API_KEY)
    python scripts/gemini_direct_image_description.py path/to/image.png
    python scripts/gemini_direct_image_description.py path/to/image.png --mode developer
    
    # Vertex AI (requires GCP authentication)
    python scripts/gemini_direct_image_description.py path/to/image.png --mode vertex
    
    # Process multiple images
    python scripts/gemini_direct_image_description.py image1.png image2.png image3.png
    
    # Save output to file
    python scripts/gemini_direct_image_description.py path/to/image.png --output description.txt
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Optional
from io import BytesIO

from dotenv import load_dotenv
from PIL import Image

# Load environment
load_dotenv()

# Model configuration - same as used in batch processing
MODEL_NAME_DEVELOPER = "gemini-2.0-flash-exp"
MODEL_NAME_VERTEX = "gemini-2.0-flash-001"

# Default prompt - same as in 3a_prepare_image_batches.py
DEFAULT_PROMPT = """Provide a terse, factual description of this image.

Report exactly what is shown:
- Content type (chart, table, diagram, etc.)
- All visible text, labels, and legends
- All numerical data, values, and units
- Axis labels and scales

Do not interpret or analyze. State only what is directly visible. Be concise."""

DEFAULT_PROMPT = """
Provide a terse, factual description of this image as structured JSON.

Rules:
- Report only what is directly visible.
- Do not interpret or analyze.
- Use one JSON object per distinct data type present (e.g., text, table, chart, diagram, photo, etc.).
- Each object should contain only fields relevant to that type.
- Use arrays if multiple similar elements exist.
- Include all visible text, labels, numbers, and units.
- Keep keys self-explanatory and values literal.

Examples of possible objects:
{
  "type": "table",
  "headers": ["Year", "Revenue ($M)", "Growth (%)"],
  "rows": [
    ["2021", "50", "10"],
    ["2022", "55", "10"],
    ["2023", "60", "9"]
  ]
}

{
  "type": "chart",
  "title": "Monthly Sales",
  "x_axis": {"label": "Month", "values": ["Jan", "Feb", "Mar"]},
  "y_axis": {"label": "Sales ($)", "values": [100, 120, 150]},
  "series": [{"label": "Product A", "values": [100, 120, 150]}]
}

{
  "type": "text",
  "content": "Warning: Battery low (15%)",
  "position": "top-right"
}

Output only valid JSON objects, separated by newlines if more than one.
Do not include prose or commentary.
"""



# -------------------------------------------------------------------
# Image Processing (copied from 3a_prepare_image_batches.py)
# -------------------------------------------------------------------

def encode_image_base64(image_path: str, max_size: int = 1024, quality: int = 85) -> str:
    """
    Encode image file as base64 string with compression.
    
    Args:
        image_path: Path to image file
        max_size: Maximum dimension (width or height) in pixels
        quality: JPEG quality (1-100, default 85)
    
    Returns:
        Base64 encoded string
    """
    try:
        # Open image
        img = Image.open(image_path)
        
        # Convert to RGB if necessary (for JPEG)
        if img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize if too large
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Compress to JPEG in memory
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=quality, optimize=True)
        buffer.seek(0)
        
        # Encode to base64
        return base64.b64encode(buffer.read()).decode("utf-8")
    
    except Exception as e:
        print(f"⚠️  Failed to compress {image_path}: {e}")
        print("    Falling back to original encoding...")
        # Fallback to original encoding
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


# -------------------------------------------------------------------
# API Calls
# -------------------------------------------------------------------

def call_gemini_developer_api(
    image_base64: str,
    prompt: str = DEFAULT_PROMPT,
    system_instruction: Optional[str] = None
) -> str:
    """
    Call Gemini Developer API directly.
    
    Args:
        image_base64: Base64 encoded image
        prompt: Text prompt for the model
        system_instruction: Optional system instruction
    
    Returns:
        Model response text
    """
    import requests
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in environment")
    
    # Build request
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME_DEVELOPER}:generateContent"
    
    headers = {
        "Content-Type": "application/json",
    }
    
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_base64
                        }
                    },
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.0
        }
    }
    
    # Add system instruction if provided
    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
    
    # Make request
    response = requests.post(
        url,
        headers=headers,
        params={"key": api_key},
        json=payload,
        timeout=60
    )
    
    if response.status_code != 200:
        raise RuntimeError(f"API call failed: {response.status_code} - {response.text}")
    
    result = response.json()
    
    # Extract text from response
    try:
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected response format: {e}\n{json.dumps(result, indent=2)}") from e


def call_gemini_vertex_api(
    image_base64: str,
    prompt: str = DEFAULT_PROMPT,
    system_instruction: Optional[str] = None
) -> str:
    """
    Call Gemini via Vertex AI.
    
    Args:
        image_base64: Base64 encoded image
        prompt: Text prompt for the model
        system_instruction: Optional system instruction
    
    Returns:
        Model response text
    """
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel, Part
    except ImportError as exc:
        raise ImportError(
            "Vertex AI SDK not installed. Install with: pip install google-cloud-aiplatform"
        ) from exc
    
    project = os.environ.get("GCP_PROJECT")
    location = os.environ.get("GCP_LOCATION", "us-central1")
    
    if not project:
        raise ValueError("GCP_PROJECT not set in environment")
    
    # Initialize Vertex AI
    vertexai.init(project=project, location=location)
    
    # Create model
    model = GenerativeModel(MODEL_NAME_VERTEX)
    
    # Prepare content
    image_part = Part.from_data(
        data=base64.b64decode(image_base64),
        mime_type="image/jpeg"
    )
    
    text_part = Part.from_text(prompt)
    
    # Generate content
    generation_config = {
        "temperature": 0.0,
    }
    
    # Build request
    contents = [image_part, text_part]
    
    # Add system instruction if provided
    kwargs = {"generation_config": generation_config}
    if system_instruction:
        kwargs["system_instruction"] = system_instruction
    
    response = model.generate_content(contents, **kwargs)
    
    return response.text


# -------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------

def describe_image(
    image_path: Path,
    mode: str = "developer",
    prompt: str = DEFAULT_PROMPT,
    system_instruction: Optional[str] = None,
    output_path: Optional[Path] = None
) -> str:
    """
    Get description of an image using Gemini API.
    
    Args:
        image_path: Path to image file
        mode: 'developer' or 'vertex'
        prompt: Text prompt for the model
        system_instruction: Optional system instruction
        output_path: Optional path to save output
    
    Returns:
        Description text
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    model_name = MODEL_NAME_DEVELOPER if mode == "developer" else MODEL_NAME_VERTEX
    print(f"📷 Processing: {image_path.name}")
    print(f"🔧 Mode: {mode}")
    print(f"🤖 Model: {model_name}")
    
    # Encode image
    print("🔄 Encoding image...")
    image_base64 = encode_image_base64(str(image_path))
    
    # Call API
    print("🌐 Calling Gemini API...")
    if mode == "developer":
        description = call_gemini_developer_api(
            image_base64=image_base64,
            prompt=prompt,
            system_instruction=system_instruction
        )
    elif mode == "vertex":
        description = call_gemini_vertex_api(
            image_base64=image_base64,
            prompt=prompt,
            system_instruction=system_instruction
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    # Print result
    print("\n" + "=" * 60)
    print("✅ DESCRIPTION")
    print("=" * 60)
    print(description)
    print("=" * 60 + "\n")
    
    # Save to file if requested
    if output_path:
        output_path.write_text(description)
        print(f"💾 Saved to: {output_path}")
    
    return description


def main():
    parser = argparse.ArgumentParser(
        description="Get image descriptions using Gemini API (direct calls)"
    )
    parser.add_argument(
        "images",
        nargs="+",
        type=Path,
        help="Path(s) to image file(s)"
    )
    parser.add_argument(
        "--mode",
        choices=["developer", "vertex"],
        default="developer",
        help="API mode: 'developer' (default) or 'vertex'"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Custom prompt (default: use standard prompt)"
    )
    parser.add_argument(
        "--system-instruction",
        type=str,
        default=None,
        help="Optional system instruction"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (only works with single image)"
    )
    
    args = parser.parse_args()
    
    # Validate environment
    if args.mode == "developer":
        if not os.environ.get("GEMINI_API_KEY"):
            print("❌ Error: GEMINI_API_KEY not set")
            print("Set it in .env or environment:")
            print("  export GEMINI_API_KEY='your-api-key'")
            return 1
    elif args.mode == "vertex":
        if not os.environ.get("GCP_PROJECT"):
            print("❌ Error: GCP_PROJECT not set")
            print("Set it in .env or environment:")
            print("  export GCP_PROJECT='your-project-id'")
            return 1
    
    # Handle output file validation
    if args.output and len(args.images) > 1:
        print("❌ Error: --output only works with a single image")
        return 1
    
    print("\n🚀 Gemini Direct Image Description")
    print("=" * 60)
    
    # Process images
    results = []
    for image_path in args.images:
        try:
            description = describe_image(
                image_path=image_path,
                mode=args.mode,
                prompt=args.prompt,
                system_instruction=args.system_instruction,
                output_path=args.output if len(args.images) == 1 else None
            )
            results.append((image_path, description))
        except Exception as e:
            print(f"❌ Error processing {image_path.name}: {e}")
            continue
    
    # Summary
    if len(args.images) > 1:
        print("\n" + "=" * 60)
        print(f"✅ Processed {len(results)}/{len(args.images)} images successfully")
        print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

