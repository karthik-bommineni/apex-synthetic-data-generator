"""Send the master YAML to any OpenAI-compatible chat model and validate the result."""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import yaml

from apex.schema import Spec, SpecError, parse_spec


def strip_yaml_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    lines = lines[1:]
    if lines and lines[0].strip().lower() in {"yaml", "yml"}:
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def populate(master_path: str | Path, output_path: str | Path) -> Spec:
    """Fill a master YAML using OPENAI_BASE_URL, OPENAI_API_KEY, and APEX_MODEL."""
    source = Path(master_path)
    destination = Path(output_path)
    master_text = source.read_text(encoding="utf-8")
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SpecError(
            "OPENAI_API_KEY is not set. Paste specs/master.yaml into any chat model, "
            "or set OPENAI_API_KEY, OPENAI_BASE_URL, and APEX_MODEL."
        )
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("APEX_MODEL", "gpt-4o-mini")
    prompt = (
        "The YAML below contains the instructions, the user request, and the skeleton. "
        "Follow the instructions exactly. Return only the populated specification as YAML, "
        "with no markdown fence and no commentary.\n\n"
        + master_text
    )
    try:
        response = httpx.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "temperature": 0.2,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You populate synthetic-data specifications. "
                            "Return YAML only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=180.0,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise SpecError(f"LLM request failed: {exc}") from exc
    payload = response.json()
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise SpecError("LLM response did not include a message") from exc
    yaml_text = strip_yaml_fence(str(content))
    try:
        loaded = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        raise SpecError(f"LLM output is not valid YAML: {exc}") from exc
    if not isinstance(loaded, dict):
        raise SpecError("LLM output must be a YAML mapping")
    spec = parse_spec(loaded)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml_text.rstrip() + "\n", encoding="utf-8")
    return spec
