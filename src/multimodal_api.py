"""Provide the AURA OS core engine for generative and multimodal workflows.

This module exposes GPT-4o dashboard generation, Whisper-powered voice search,
and CLIP-based image-to-catalog matching. Its DALL-E 3 text-to-design endpoint
accepts a ``{prompt, color}`` request and returns two image URLs. At the default
standard-quality estimate, the total cost for both images is $0.08.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

router = APIRouter()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CATALOG = PROJECT_ROOT / "Dataset" / "Superstore.csv"
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
_clip_model: Any = None
_clip_processor: Any = None


class DashboardRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=500)


class DesignRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=500)
    color: str = Field(min_length=1, max_length=40)


def _catalog(path: Path = DEFAULT_CATALOG) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Product catalog was not found: {path}")
    dataframe = pd.read_csv(path)
    dataframe.columns = [str(column).strip() for column in dataframe.columns]
    if dataframe.empty:
        raise ValueError("The product catalog contains no rows.")
    return dataframe


def _generate_design_image(prompt: str, variation: str) -> str:
    client = _openai_client()
    response = client.images.generate(
        model="dall-e-3",
        prompt=f"{prompt}. {variation}. Indian ethnic wear, studio lighting, 4k.",
        size="1024x1024",
        quality="standard",
        n=1,
    )
    image_url = response.data[0].url
    if not image_url:
        raise RuntimeError("The image service returned no URL.")
    return image_url


def _design_price_estimate() -> float:
    try:
        price_per_image = float(os.getenv("AURA_DALLE3_STANDARD_IMAGE_PRICE_USD", "0.04"))
    except ValueError as exc:
        raise ValueError("AURA_DALLE3_STANDARD_IMAGE_PRICE_USD must be a number.") from exc
    if price_per_image < 0:
        raise ValueError("AURA_DALLE3_STANDARD_IMAGE_PRICE_USD cannot be negative.")
    return round(price_per_image * 2, 2)


def _openai_client() -> Any:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY must be configured on the API server.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the openai package to enable GPT-4o and Whisper features.") from exc
    return OpenAI(api_key=api_key)


def _dashboard_spec(prompt: str) -> dict[str, Any]:
    client = _openai_client()
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Create a useful retail dashboard specification from the user request. "
                    "Return JSON only with keys title, subtitle, metrics, chart, and insights. "
                    "metrics must be an array of at most 4 objects with label, value, and change. "
                    "chart must have title, labels, and values arrays with the same length, at most 8. "
                    "insights must be an array of at most 3 short strings. "
                    "Use plausible demo values and label them as illustrative; do not imply access to live data. "
                    "Do not return executable code, HTML, or scripts."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content or "{}"
    data = json.loads(content)
    metrics = data.get("metrics", [])
    chart = data.get("chart", {})
    labels = chart.get("labels", [])
    values = chart.get("values", [])

    dashboard = {
        "title": str(data.get("title", "Your retail dashboard"))[:100],
        "subtitle": str(data.get("subtitle", "Illustrative view generated from your brief."))[:180],
        "metrics": [
            {
                "label": str(item.get("label", "Metric"))[:50],
                "value": str(item.get("value", "--"))[:40],
                "change": str(item.get("change", "Illustrative"))[:60],
            }
            for item in metrics[:4]
            if isinstance(item, dict)
        ],
        "chart": {
            "title": str(chart.get("title", "Performance overview"))[:80],
            "labels": [str(label)[:40] for label in labels[:8]],
            "values": [max(0, min(1000000000, float(value))) for value in values[:8]],
        },
        "insights": [str(insight)[:180] for insight in data.get("insights", [])[:3]],
    }
    paired_count = min(len(dashboard["chart"]["labels"]), len(dashboard["chart"]["values"]))
    dashboard["chart"]["labels"] = dashboard["chart"]["labels"][:paired_count]
    dashboard["chart"]["values"] = dashboard["chart"]["values"][:paired_count]
    serialized = json.dumps(dashboard, ensure_ascii=True)
    jsx = f"const dashboard = {serialized};\n<GeneratedDashboard data={{dashboard}} />"
    return {"dashboard": dashboard, "jsx": jsx}


@router.post("/api/generate-design")
async def generate_design(payload: DesignRequest) -> dict[str, Any]:
    """Generate two ethnic-wear design concepts from a prompt and color."""
    prompt = payload.prompt.strip()
    color = payload.color.strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="A design prompt is required.")
    if not color:
        raise HTTPException(status_code=422, detail="A design color is required.")

    design_prompt = f"{prompt}. Dominant color: {color}"
    try:
        image_url_1, image_url_2 = await asyncio.gather(
            asyncio.to_thread(_generate_design_image, design_prompt, "Design variation one"),
            asyncio.to_thread(_generate_design_image, design_prompt, "Design variation two"),
        )
        return {
            "image_url_1": image_url_1,
            "image_url_2": image_url_2,
            "estimated_price": _design_price_estimate(),
        }
    except EnvironmentError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Design generation failed: {exc}") from exc


def _search_products(query: str, limit: int = 8) -> list[dict[str, Any]]:
    dataframe = _catalog()
    product_column = next((column for column in ("Product Name", "Product", "Item") if column in dataframe.columns), None)
    if product_column is None:
        raise ValueError("Catalog must contain a Product Name column.")

    sales_column = next((column for column in ("Sales", "Sales_INR") if column in dataframe.columns), None)
    category_column = next((column for column in ("Category", "Sub-Category") if column in dataframe.columns), None)
    query_normalized = " ".join(query.casefold().split())
    tokens = set(query_normalized.split())
    products = dataframe.drop_duplicates(subset=[product_column]).copy()
    results = []

    for _, row in products.iterrows():
        name = str(row[product_column])
        normalized_name = " ".join(name.casefold().split())
        token_score = len(tokens.intersection(normalized_name.split())) / max(len(tokens), 1)
        score = max(token_score, SequenceMatcher(None, query_normalized, normalized_name).ratio())
        if query_normalized in normalized_name:
            score = 1.0
        if score < 0.16:
            continue
        result: dict[str, Any] = {"product": name, "match_score": round(score, 3)}
        if category_column:
            result["category"] = str(row[category_column])
        if sales_column:
            result["sales"] = float(pd.to_numeric(row[sales_column], errors="coerce") or 0)
        results.append(result)

    return sorted(results, key=lambda item: item["match_score"], reverse=True)[:limit]


@router.post("/api/generative-ui")
async def generate_dashboard(payload: DashboardRequest) -> dict[str, Any]:
    """Generate a validated dashboard specification and JSX source for the React renderer."""
    try:
        return await asyncio.to_thread(_dashboard_spec, payload.prompt.strip())
    except EnvironmentError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Dashboard generation failed: {exc}") from exc


@router.post("/api/voice-search")
async def voice_search(audio: UploadFile = File(...)) -> dict[str, Any]:
    """Transcribe an audio query with Whisper and match it against the product catalog."""
    content = await audio.read(MAX_UPLOAD_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="Upload a non-empty audio recording.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Audio uploads must be 12 MB or smaller.")
    try:
        client = _openai_client()
        filename = Path(audio.filename or "voice-query.webm").name
        transcript = await asyncio.to_thread(
            lambda: client.audio.transcriptions.create(
                model="whisper-1",
                file=(filename, io.BytesIO(content), audio.content_type or "audio/webm"),
            ).text
        )
        products = await asyncio.to_thread(_search_products, transcript)
        return {"transcript": transcript, "products": products}
    except EnvironmentError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Voice search failed: {exc}") from exc


def _load_clip() -> tuple[Any, Any]:
    global _clip_model, _clip_processor
    if _clip_model is None or _clip_processor is None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError("Install torch and transformers to enable CLIP image search.") from exc

        model_name = "openai/clip-vit-base-patch32"
        _clip_processor = CLIPProcessor.from_pretrained(model_name)
        _clip_model = CLIPModel.from_pretrained(model_name).to("cpu").eval()
    return _clip_model, _clip_processor


def _image_matches(image_bytes: bytes) -> list[dict[str, Any]]:
    from PIL import Image
    import torch

    dataframe = _catalog()
    product_column = next((column for column in ("Product Name", "Product", "Item") if column in dataframe.columns), None)
    if product_column is None:
        raise ValueError("Catalog must contain a Product Name column.")

    products = dataframe.drop_duplicates(subset=[product_column]).head(500).copy()
    category_column = next((column for column in ("Category", "Sub-Category") if column in dataframe.columns), None)
    sales_column = next((column for column in ("Sales", "Sales_INR") if column in dataframe.columns), None)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    model, processor = _load_clip()
    image_inputs = processor(images=image, return_tensors="pt")
    with torch.inference_mode():
        image_embedding = model.get_image_features(**image_inputs)
        image_embedding = torch.nn.functional.normalize(image_embedding, dim=-1)

        descriptions = []
        for _, row in products.iterrows():
            details = [str(row[product_column])]
            if category_column:
                details.append(str(row[category_column]))
            descriptions.append(" ".join(details))

        text_embeddings = []
        for start in range(0, len(descriptions), 32):
            text_inputs = processor(
                text=descriptions[start : start + 32],
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            text_embedding = model.get_text_features(**text_inputs)
            text_embeddings.append(torch.nn.functional.normalize(text_embedding, dim=-1))
        similarities = torch.cat(text_embeddings, dim=0) @ image_embedding.T
        top_indices = torch.topk(similarities.squeeze(-1), k=min(8, len(products))).indices.tolist()

    matches = []
    for index in top_indices:
        row = products.iloc[index]
        match: dict[str, Any] = {
            "product": str(row[product_column]),
            "similarity": round(float(similarities[index].item()), 4),
        }
        if category_column:
            match["category"] = str(row[category_column])
        if sales_column:
            match["sales"] = float(pd.to_numeric(row[sales_column], errors="coerce") or 0)
        matches.append(match)
    return matches


@router.post("/api/image-search")
async def image_search(image: UploadFile = File(...)) -> dict[str, Any]:
    """Find catalog product descriptions most similar to an uploaded image using CLIP."""
    content = await image.read(MAX_UPLOAD_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="Upload a non-empty image.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image uploads must be 12 MB or smaller.")
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="The uploaded file must be an image.")
    try:
        products = await asyncio.to_thread(_image_matches, content)
        return {"products": products, "model": "openai/clip-vit-base-patch32"}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Image search failed: {exc}") from exc
