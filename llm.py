"""Render extraction messages and record simulated or live API calls."""

from copy import deepcopy
import json
import os
from pathlib import Path
import re
from time import perf_counter

from dotenv import dotenv_values
from openai import APIError, OpenAI


DEFAULT_MAX_INPUT_TOKENS = 120_000
DEFAULT_MAX_OUTPUT_TOKENS = 4_096
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"
DEFAULT_MAX_ROUNDS = 10
DEFAULT_MAX_TOTAL_INPUT_TOKENS = 250_000
API_URLS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "openai": "https://api.openai.com/v1",
}
INPUT_ESTIMATOR = "ceil(UTF-8 bytes of all rendered message roles and content / 4)"
CHAT_INPUT_ESTIMATOR = "ceil(UTF-8 bytes of the complete serialized request payload / 4)"
USAGE_FIELDS = ("input_tokens", "output_tokens", "total_tokens", "cached_tokens", "reasoning_tokens")


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


def aggregate_usage(calls):
    """Aggregate reported usage while keeping any missing contribution unknown."""
    reports = [call.get("usage") for call in calls if call["api_attempts"]]
    if not any(report is not None for report in reports):
        return None
    totals = {}
    for field in USAGE_FIELDS:
        values = [report.get(field) if report else None for report in reports]
        totals[field] = sum(values) if all(value is not None for value in values) else None
    return totals


def _gemini_result(response, call, tools_enabled=False):
    usage = response.get("usage")
    call.update(
        reported_model=response.get("model"),
        response_id=response.get("id"),
        usage=None if not isinstance(usage, dict) else {
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        },
    )
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        call.update(status="provider_error", error={"message": "Gemini returned no completion choices."})
        return None

    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    call["response_status"] = finish_reason
    message = choice.get("message")
    if message is None and finish_reason in {"content_filter", "length"}:
        message = {}
    if not isinstance(message, dict) or (
        message.get("content") is not None and not isinstance(message["content"], str)
    ):
        call.update(status="provider_error", error={"message": "Gemini returned an invalid assistant message."})
        return None
    refusal = message.get("refusal")
    if refusal or finish_reason == "content_filter":
        call.update(status="refused", error={"reason": finish_reason, "refusals": [refusal] if refusal else []})
    elif finish_reason == "length":
        call.update(status="incomplete", error={"reason": "max_output_tokens"})
    elif tools_enabled and message.get("tool_calls") and finish_reason in {"tool_calls", "stop", None}:
        call["status"] = "tool_calls"
    elif finish_reason == "stop":
        call["status"] = "completed"
    else:
        call.update(status="provider_error", error={"reason": finish_reason})
    return message.get("content")


class LLM:
    def __init__(
        self, *, model=None, api_key=None, simulated_response=None, provider="openai",
        max_input_tokens=DEFAULT_MAX_INPUT_TOKENS, max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
        max_rounds=DEFAULT_MAX_ROUNDS, max_total_input_tokens=DEFAULT_MAX_TOTAL_INPUT_TOKENS,
    ):
        self.model = model
        self.provider = provider
        self._api_key = api_key
        self.simulated_response = simulated_response
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.max_rounds = max_rounds
        self.max_total_input_tokens = max_total_input_tokens
        self.accepted_input_tokens = 0
        self.calls = []
        self.method_trace = {}
        self._assistant_history = {}

    def _safe_error(self, message):
        if message is None:
            return None
        message = str(message)
        if self._api_key:
            message = message.replace(self._api_key, "[redacted]")
        return re.sub(r"\b(?:sk-|AIza)[^\s\"',;]+", "[redacted]", message)

    def _record_call(self, messages, byte_count, estimate):
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
        return call

    def _request(self, call, payload):
        print(
            f"Live {self.provider} request for {self.model} | {call['input_utf8_bytes']:,} rendered UTF-8 bytes | "
            f"{call['estimated_input_tokens']:,} estimated input tokens (limit {self.max_input_tokens:,}) | "
            f"maximum output {self.max_output_tokens:,} tokens",
            flush=True,
        )
        with OpenAI(api_key=self._api_key, base_url=API_URLS[self.provider], max_retries=0) as client:
            started = perf_counter()
            call["api_attempts"] = 1
            try:
                if self.provider == "gemini":
                    response = client.chat.completions.with_raw_response.create(**payload)
                    body = response.http_response.json()
                    if not isinstance(body, dict):
                        raise ValueError("Expected a JSON object from Gemini.")
                    return body
                return client.responses.create(**payload)
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
            except ValueError as error:
                call.update(status="provider_error", error={"message": self._safe_error(error)})
                return None
            finally:
                call["latency_seconds"] = perf_counter() - started

    def _restore_assistant_history(self, messages):
        for index, message in enumerate(messages):
            original = self._assistant_history.get(index)
            if message.get("role") != "assistant" or not original:
                continue
            if "extra_content" in original:
                message["extra_content"] = deepcopy(original["extra_content"])
            by_id = {tool.get("id"): tool for tool in original.get("tool_calls") or []}
            for position, tool in enumerate(message.get("tool_calls") or []):
                if tool.get("id") in by_id:
                    message["tool_calls"][position] = deepcopy(by_id[tool["id"]])

    def chat(self, messages, tools=None, *, simulate=None):
        """Send a bounded Gemini turn and return the original Chat Completions envelope."""
        if self.provider != "gemini":
            raise ValueError("Agentic chat is supported only for Gemini.")
        if self.simulated_response is not None and simulate is None:
            raise ValueError("A simulated chat requires scripted responses.")
        sent = deepcopy(messages)
        self._restore_assistant_history(sent)
        payload = {"model": self.model, "messages": sent, "max_tokens": self.max_output_tokens}
        if tools is not None:
            payload.update(tools=deepcopy(tools), tool_choice="auto")
        byte_count = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        estimate = (byte_count + 3) // 4
        call = self._record_call(sent, byte_count, estimate)
        proposed_total = self.accepted_input_tokens + estimate
        call.update(
            round=len(self.calls), request=payload, max_total_input_tokens=self.max_total_input_tokens,
            cumulative_estimated_input_tokens=self.accepted_input_tokens,
        )
        if call["round"] > self.max_rounds:
            call.update(status="max_rounds_reached", error={"message": "The maximum number of rounds was reached."})
        elif call["would_exceed_input_limit"]:
            call.update(status="input_too_large", error={
                "message": f"Estimated input of {estimate:,} exceeds the {self.max_input_tokens:,}-token request limit."
            })
        elif proposed_total > self.max_total_input_tokens:
            call.update(status="total_input_budget_exceeded", error={
                "message": f"Estimated cumulative input of {proposed_total:,} exceeds the "
                           f"{self.max_total_input_tokens:,}-token limit."
            })
        else:
            self.accepted_input_tokens = proposed_total
            call.update(input_accepted=True, cumulative_estimated_input_tokens=proposed_total,
                        simulated=simulate is not None)
            response = simulate(sent) if simulate is not None else self._request(call, payload)
            if response is None:
                return None
            call["raw_response"] = _gemini_result(response, call, tools_enabled=tools is not None)
            choices = response.get("choices") or []
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                call["response_message"] = deepcopy(choices[0].get("message"))
            if call["status"] in {"completed", "tool_calls"}:
                self._assistant_history[len(sent)] = deepcopy(call["response_message"])
                return response
        return None

    def invoke(self, prompt, state):
        messages = render_messages(prompt, state)
        byte_count, estimate = measure_input(messages)
        call = self._record_call(messages, byte_count, estimate)
        if self.simulated_response is not None:
            call["status"] = "dry_run"
            return self.simulated_response

        if estimate > self.max_input_tokens:
            call["status"] = "input_too_large"
            call["error"] = {
                "message": f"Estimated input of {estimate:,} tokens exceeds the {self.max_input_tokens:,}-token limit."
            }
            return None

        if self.provider == "gemini":
            response = self._request(call, {"model": self.model, "messages": messages, "max_tokens": self.max_output_tokens})
            return _gemini_result(response, call) if response is not None else None

        sampling = _sampling_settings(self.model)
        call["sampling_settings"] = sampling
        response = self._request(call, {
            "model": self.model, "input": messages, "max_output_tokens": self.max_output_tokens,
            "truncation": "disabled", "store": False, **sampling,
        })
        if response is None:
            return None

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
