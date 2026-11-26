"""Render extraction messages and record simulated or live API calls."""

import os
from pathlib import Path
import re
from time import perf_counter

from dotenv import dotenv_values
from openai import APIError, OpenAI


DEFAULT_MAX_INPUT_TOKENS = 120_000
DEFAULT_MAX_OUTPUT_TOKENS = 4_096
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"
API_URLS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "openai": "https://api.openai.com/v1",
}
INPUT_ESTIMATOR = "ceil(UTF-8 bytes of all rendered message roles and content / 4)"


def live_settings(env_path: Path, model_override=None, provider="openai"):
    """Resolve the selected provider's configuration without changing .env."""
    file_values = dotenv_values(env_path, interpolate=False)

    def first_value(source, *keys):
        return next((source[key].strip() for key in keys if source.get(key) and source[key].strip()), None)

    if provider == "gemini":
        model_variable = "GEMINI_MODEL"
        key_names = ("GEMINI_KEY",)
        default_model = DEFAULT_GEMINI_MODEL
    else:
        model_variable = "OPENAI_MODEL"
        key_names = ("OPENAI_API_KEY", "OPEN_AI_KEY")
        default_model = None
    model = model_override or first_value(os.environ, model_variable) or first_value(
        file_values, model_variable
    ) or default_model
    if not model or not model.strip():
        raise ValueError(f"Configure {model_variable} or pass --model for a live run.")
    key = first_value(os.environ, *key_names) or first_value(file_values, *key_names)
    if not key:
        raise ValueError(f"Configure {' or '.join(key_names)} for a live run.")
    return model.strip(), key


def render_messages(prompt, state):
    templates, fixed, state_keys, output_type = prompt
    if output_type != "str":
        raise ValueError("The extraction client expects a raw string response.")
    parameters = {**fixed, **{key: state[key] for key in state_keys}}
    return [{"role": role, "content": template.format_map(parameters)} for role, template in templates]


def measure_input(messages):
    byte_count = sum(len(message[key].encode("utf-8")) for message in messages for key in ("role", "content"))
    return byte_count, (byte_count + 3) // 4


def _sampling_settings(model):
    snapshot = r"(?:-\d{4}-\d{2}-\d{2})?"
    if re.fullmatch(r"(?:gpt-4o(?:-mini)?|gpt-4\.1(?:-mini|-nano)?)" + snapshot, model):
        return {"temperature": 0}
    if re.fullmatch(r"gpt-5\.[12]" + snapshot, model):
        return {"temperature": 0, "reasoning": {"effort": "none"}}
    return {}


def _usage(response):
    usage = response.usage
    if usage is None:
        return None
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "cached_tokens": getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", None),
        "reasoning_tokens": getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", None),
    }


def _gemini_result(response, call):
    usage = getattr(response, "usage", None)
    call.update(
        reported_model=getattr(response, "model", None),
        response_id=getattr(response, "id", None),
        usage=None if usage is None else {
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "cached_tokens": getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None),
            "reasoning_tokens": getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None),
        },
    )
    if not response.choices:
        call.update(status="provider_error", error={"message": "Gemini returned no completion choices."})
        return None

    choice = response.choices[0]
    finish_reason = choice.finish_reason
    message = getattr(choice, "message", None)
    refusal = getattr(message, "refusal", None)
    call["response_status"] = finish_reason
    if refusal or finish_reason == "content_filter":
        call.update(status="refused", error={"reason": finish_reason, "refusals": [refusal] if refusal else []})
    elif finish_reason == "length":
        call.update(status="incomplete", error={"reason": "max_output_tokens"})
    elif finish_reason == "stop":
        call["status"] = "completed"
    else:
        call.update(status="provider_error", error={"reason": finish_reason})
    return getattr(message, "content", None)


class LLM:
    def __init__(
        self, *, model=None, api_key=None, simulated_response=None, provider="openai",
        max_input_tokens=DEFAULT_MAX_INPUT_TOKENS, max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
    ):
        self.model = model
        self.provider = provider
        self._api_key = api_key
        self.simulated_response = simulated_response
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.calls = []

    def _safe_error(self, message):
        if message is None:
            return None
        message = str(message)
        if self._api_key:
            message = message.replace(self._api_key, "[redacted]")
        return re.sub(r"\b(?:sk-|AIza)[^\s\"',;]+", "[redacted]", message)

    def invoke(self, prompt, state):
        messages = render_messages(prompt, state)
        byte_count, estimate = measure_input(messages)
        call = {
            "messages": messages,
            "input_utf8_bytes": byte_count,
            "estimated_input_tokens": estimate,
            "max_input_tokens": self.max_input_tokens,
            "would_exceed_input_limit": estimate > self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "provider": self.provider,
            "requested_model": self.model,
            "reported_model": None,
            "response_status": None,
            "api_attempts": 0,
            "latency_seconds": None,
            "usage": None,
            "error": None,
            "sampling_settings": {},
        }
        self.calls.append(call)
        if self.simulated_response is not None:
            call["status"] = "dry_run"
            return self.simulated_response

        if estimate > self.max_input_tokens:
            call["status"] = "input_too_large"
            call["error"] = {
                "message": f"Estimated input of {estimate:,} tokens exceeds the {self.max_input_tokens:,}-token limit."
            }
            return None

        print(
            f"Live {self.provider} request for {self.model} | {byte_count:,} rendered UTF-8 bytes | "
            f"{estimate:,} estimated input tokens (limit {self.max_input_tokens:,}) | "
            f"maximum output {self.max_output_tokens:,} tokens",
            flush=True,
        )
        gemini = self.provider == "gemini"
        sampling = {} if gemini else _sampling_settings(self.model)
        call["sampling_settings"] = sampling
        with OpenAI(api_key=self._api_key, base_url=API_URLS[self.provider], max_retries=0) as client:
            started = perf_counter()
            call["api_attempts"] = 1
            try:
                if gemini:
                    response = client.chat.completions.create(
                        model=self.model, messages=messages, max_tokens=self.max_output_tokens,
                    )
                else:
                    response = client.responses.create(
                        model=self.model,
                        input=messages,
                        max_output_tokens=self.max_output_tokens,
                        truncation="disabled",
                        store=False,
                        **sampling,
                    )
            except APIError as error:
                call["status"] = "provider_error"
                call["error"] = {
                    "type": type(error).__name__,
                    "message": self._safe_error(error.message),
                    "code": self._safe_error(getattr(error, "code", None)),
                    "http_status": getattr(error, "status_code", None),
                    "request_id": self._safe_error(getattr(error, "request_id", None)),
                }
                return None
            finally:
                call["latency_seconds"] = perf_counter() - started

        if gemini:
            return _gemini_result(response, call)

        call.update(
            reported_model=response.model,
            response_status=response.status,
            usage=_usage(response),
            response_id=response.id,
        )
        if response.status == "incomplete":
            call["status"] = "incomplete"
            call["error"] = {"reason": getattr(response.incomplete_details, "reason", None)}
        elif response.status != "completed":
            call["status"] = "provider_error"
            call["error"] = {
                "message": self._safe_error(getattr(response.error, "message", None)) or f"Response status {response.status}",
                "code": self._safe_error(getattr(response.error, "code", None)),
            }
        else:
            refusals = [
                content.refusal
                for item in response.output if item.type == "message"
                for content in item.content if content.type == "refusal"
            ]
            call["status"] = "refused" if refusals else "completed"
            if refusals:
                call["error"] = {"refusals": refusals}
        return response.output_text
