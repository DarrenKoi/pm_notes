#!/usr/bin/env python3
"""Send a local image file to an OpenAI-compatible vLLM endpoint."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
from pathlib import Path
from typing import Any

import requests


DEFAULT_PROMPT = "Describe the image briefly and focus on the visible UI."


def normalize_api_base_url(base_url: str) -> str:
    base = base_url.rstrip("/")

    if base.endswith("/chat/completions"):
        return base[: -len("/chat/completions")]

    if base.endswith("/models"):
        return base[: -len("/models")]

    if base.endswith("/v1"):
        return base

    return f"{base}/v1"


def build_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def http_json(
    url: str,
    *,
    headers: dict[str, str],
    timeout: int,
    verify_ssl: bool,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        if payload is None:
            response = requests.get(
                url,
                headers=headers,
                timeout=timeout,
                verify=verify_ssl,
            )
        else:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout,
                verify=verify_ssl,
            )
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = exc.response.text if exc.response is not None else ""
        status = exc.response.status_code if exc.response is not None else "?"
        raise RuntimeError(f"HTTP {status} for {url}\n{body}") from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"Request failed for {url}: {exc}") from exc

    try:
        return response.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Non-JSON response from {url}\n{response.text}") from exc


def list_models(
    api_base_url: str,
    *,
    api_key: str | None,
    timeout: int,
    verify_ssl: bool,
) -> list[dict[str, Any]]:
    payload = http_json(
        f"{api_base_url}/models",
        headers=build_headers(api_key),
        timeout=timeout,
        verify_ssl=verify_ssl,
    )
    models = payload.get("data")
    if not isinstance(models, list):
        raise RuntimeError("The /v1/models response did not contain a model list.")
    return models


def resolve_model_name(
    api_base_url: str,
    *,
    model_name: str | None,
    api_key: str | None,
    timeout: int,
    verify_ssl: bool,
) -> str:
    if model_name:
        return model_name

    models = list_models(
        api_base_url,
        api_key=api_key,
        timeout=timeout,
        verify_ssl=verify_ssl,
    )

    for item in models:
        candidate = item.get("id")
        if isinstance(candidate, str) and candidate:
            return candidate

    raise RuntimeError("No model id was found in /v1/models.")


def image_file_to_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{image_b64}"


def extract_text_content(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        return json.dumps(response_json, ensure_ascii=False, indent=2)

    message = choices[0].get("message", {})
    content = message.get("content")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        if text_parts:
            return "\n".join(text_parts)

    return json.dumps(response_json, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a local image to an OpenAI-compatible vLLM endpoint.",
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="Proxy or server root URL, for example http://host/proxy/8001",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Local image path. Not required when --list-models is used.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Instruction or question for the model.",
    )
    parser.add_argument(
        "--model",
        help="Served model name. If omitted, the first model from /v1/models is used.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("VLLM_API_KEY"),
        help="Optional API key. Defaults to the VLLM_API_KEY environment variable.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=300,
        help="Maximum output tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Request timeout in seconds.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List models from /v1/models and exit.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print the full JSON response instead of only the text output.",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        help="Optional path to save the raw JSON response.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_base_url = normalize_api_base_url(args.base_url)
    verify_ssl = not args.insecure

    if args.list_models:
        models = list_models(
            api_base_url,
            api_key=args.api_key,
            timeout=args.timeout,
            verify_ssl=verify_ssl,
        )
        print(json.dumps(models, ensure_ascii=False, indent=2))
        return 0

    if args.image is None:
        raise SystemExit("--image is required unless --list-models is used.")

    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        raise SystemExit(f"Image not found: {image_path}")

    model_name = resolve_model_name(
        api_base_url,
        model_name=args.model,
        api_key=args.api_key,
        timeout=args.timeout,
        verify_ssl=verify_ssl,
    )

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": args.prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_file_to_data_url(image_path)},
                    },
                ],
            }
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    response_json = http_json(
        f"{api_base_url}/chat/completions",
        headers=build_headers(args.api_key),
        timeout=args.timeout,
        verify_ssl=verify_ssl,
        payload=payload,
    )

    if args.save_json:
        args.save_json.write_text(
            json.dumps(response_json, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"API base URL: {api_base_url}", file=sys.stderr)
    print(f"Model: {model_name}", file=sys.stderr)

    if args.raw:
        print(json.dumps(response_json, ensure_ascii=False, indent=2))
    else:
        print(extract_text_content(response_json))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
