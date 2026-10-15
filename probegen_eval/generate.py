"""Generate predictions with the paper's default prompts and preprocessing."""

import base64
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image

from .data import TASKS

MODELS = {  # name: (provider, model ID)
    "nano-banana-2": ("gemini", "gemini-3.1-flash-image-preview"),
    "nano-banana-pro": ("gemini", "gemini-3-pro-image-preview"),
    "nano-banana-2.1": ("gemini", "gemini-nano-banana-2.1"),
    "gpt-image-2": ("openai", "gpt-image-2"),
    "seedream-5.0": ("seedream", "doubao-seedream-5-0-260128"),
}
DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "seedream": "https://ark.cn-beijing.volces.com/api/v3",
}

PROMPTS = {
    "segmentation": "Segment {label}. Make the target object pure white and everything else pure black.",
    "reasoning": "{label}\nCreate a binary segmentation mask for the target object that satisfies "
                 "the above description. Make the target object pure white and everything else pure black.",
    "ood": "Segment all {label}. Make the target objects (one or more) pure white and everything else pure black.",
    "depth": "Create a grayscale depth map of this image: brighter = closer, darker = farther.",
    "counting": "Put one small solid red dot precisely on each {label} in this image. "
                "Use the same bright red color for every dot. Do not change anything else in the image.",
}
CLASS_NAMES = {
    "DRAM": {3: "bird", 4: "boat", 5: "bottle", 8: "cat", 9: "chair", 10: "cow",
             12: "dog", 13: "horse", 15: "person", 16: "pottedplant", 17: "sheep"},
    "Manga109": {3: "faces", 4: "characters", 5: "speech balloons", 6: "onomatopoeia"},
}


def render_prompt(dataset, sample):
    if dataset == "ReasonSeg":
        return PROMPTS["reasoning"].format(label=sample.query)
    if dataset in CLASS_NAMES:
        return PROMPTS["ood"].format(label=CLASS_NAMES[dataset][sample.target["class_id"]])
    return PROMPTS[TASKS[dataset]].format(label=sample.query)


def prepare_input(path):
    """Resize to a 1024-pixel long side and encode as JPEG (quality 95)."""
    with Image.open(path) as image:
        w, h = image.size
        size = (1024, int(h * 1024 / w)) if w >= h else (int(w * 1024 / h), 1024)
        image = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        stream = BytesIO()
        image.save(stream, "JPEG", quality=95)
    return stream.getvalue()


def load_api_settings(provider, path=Path("api.local.yaml")):
    import yaml
    settings = ((yaml.safe_load(path.read_text()) or {}) if path.is_file() else {}).get(provider) or {}
    key = str(settings.get("api_key") or "")
    if not key or key.startswith("<"):
        raise SystemExit(f"Set {provider}.api_key in {path} (copy api.sample.yaml first)")
    return settings.get("base_url", DEFAULT_BASE_URLS[provider]).rstrip("/"), key, settings.get("auth_mode")


def make_backend(provider, model, base_url, key, auth_mode):
    """Return a function (prompt, input JPEG bytes) -> generated image bytes."""
    if provider == "gemini":
        bearer = auth_mode == "bearer"
        client = httpx.Client(base_url=base_url + "/", timeout=300,
                              headers={"Authorization": f"Bearer {key}"} if bearer else {"x-goog-api-key": key})

        def gemini(prompt, jpeg):
            parts = [{"text": prompt}, {"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(jpeg).decode()}}]
            response = client.post(f"models/{model}:generateContent", json={
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": {"responseModalities": ["IMAGE"]}})
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
            for candidate in response.json().get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    if "inlineData" in part and not part.get("thought"):   # skip intermediate "thought" images
                        return base64.b64decode(part["inlineData"]["data"])
            raise RuntimeError("response contains no image")
        return gemini

    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=base_url)

    def openai_compatible(prompt, jpeg):
        if provider == "openai":
            response = client.images.edit(model=model, prompt=prompt, image=[("input.jpg", jpeg, "image/jpeg")])
        else:   # Seedream takes the input image as a data URL
            image = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
            response = client.images.generate(model=model, prompt=prompt,
                                              extra_body={"image": image, "watermark": False})
        item = response.data[0]
        return base64.b64decode(item.b64_json) if item.b64_json else httpx.get(item.url, timeout=300).raise_for_status().content
    return openai_compatible


def generate(dataset, model, samples, data_root, output_dir):
    provider, model_id = MODELS[model]
    base_url, key, auth_mode = load_api_settings(provider)
    backend = make_backend(provider, model_id, base_url, key, auth_mode)
    output_dir.mkdir(parents=True, exist_ok=True)
    failed = 0
    for index, sample in enumerate(samples, 1):
        path = output_dir / sample.prediction
        status = "exists"
        if not path.exists():
            try:
                content = backend(render_prompt(dataset, sample), prepare_input(data_root / sample.image))
                Image.open(BytesIO(content)).convert("RGB").save(path, "PNG")
                status = "ok"
            except Exception as error:
                failed += 1
                status = f"failed ({type(error).__name__}: {str(error).replace(key, '***')[:300]})"
        print(f"{dataset} {index}/{len(samples)} {sample.id}: {status}", flush=True)
    print(f"Done: {len(samples) - failed} of {len(samples)} predictions in {output_dir}"
          + (f"; rerun to retry {failed} failed" if failed else ""))
    return 1 if failed else 0
