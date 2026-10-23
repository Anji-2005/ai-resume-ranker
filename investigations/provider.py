"""Provider-supported tool calling adapter for the investigation planner."""

import json
import os
from uuid import uuid4

from investigations.schemas import ToolInvocation, ToolModelResponse


class InvestigationProviderError(RuntimeError):
    pass


def _rate(name: str) -> float | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise InvestigationProviderError(f"{name} must be numeric") from exc
    return parsed if parsed >= 0 else None


class LangChainGeminiToolProvider:
    """Gemini tool-calling adapter that returns only sanitized tool calls and usage.

    Agent execution requires configured token prices so its USD budget can fail
    closed. This does not affect the existing V2 generation provider.
    """

    def __init__(self, api_key, model_name, *, timeout_seconds=60, temperature=0, max_output_tokens=1_500):
        self.api_key = api_key
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def invoke_tools(self, prompt, tools) -> ToolModelResponse:
        input_rate = _rate("GENAI_INPUT_USD_PER_MILLION_TOKENS")
        output_rate = _rate("GENAI_OUTPUT_USD_PER_MILLION_TOKENS")
        if input_rate is None or output_rate is None:
            raise InvestigationProviderError(
                "agent cost rates must be configured before live investigation execution"
            )
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            model = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key,
                temperature=self.temperature,
                retries=0,
                request_timeout=self.timeout_seconds,
                include_thoughts=False,
                max_output_tokens=self.max_output_tokens,
            )
            response = model.bind_tools(tools, tool_choice="auto").invoke(prompt)
        except Exception as exc:
            raise InvestigationProviderError("investigation planning provider failed") from exc
        calls = []
        for item in response.tool_calls or []:
            arguments = item.get("args")
            if not isinstance(arguments, dict):
                raise InvestigationProviderError("provider returned malformed tool arguments")
            calls.append(ToolInvocation(
                id=str(item.get("id") or uuid4()), name=item.get("name", ""), arguments=arguments,
            ))
        if not calls:
            raise InvestigationProviderError("provider returned no tool call")
        usage = response.usage_metadata or {}
        input_tokens = int(usage.get("input_tokens") or usage.get("input_token_count") or max(1, len(prompt) // 4))
        output_tokens = int(usage.get("output_tokens") or usage.get("output_token_count") or max(
            1, len(json.dumps([call.model_dump() for call in calls], default=str)) // 4,
        ))
        cost = (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000
        return ToolModelResponse(
            calls=calls, input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost,
        )
