"""AWS Bedrock client wrapper for Claude Sonnet 4.5."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

from .galileo_logger import GalileoTracer

DEFAULT_MODEL_ID = "arn:aws:bedrock:us-east-1:216874796537:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0"
DEFAULT_REGION = os.environ.get("AWS_REGION", "us-east-1")


@dataclass
class BedrockClientConfig:
    """Configuration for instantiating the Bedrock runtime client."""

    model_id: str = DEFAULT_MODEL_ID
    region: str = DEFAULT_REGION
    system_prompt: str = (
        "You are a meticulous Python execution reasoning assistant. "
        "Given a program plus assertion you must reason step-by-step, then respond "
        "exactly using the requested format (e.g., [ANSWER] tags or JSON)."
    )


@dataclass
class BedrockInvocationParams:
    """Per-request inference settings."""

    reasoning_effort: str = "medium"
    max_tokens: int = 1000
    temperature: float = 0.6
    top_p: float = 0.95
    seed: Optional[int] = None
    system_prompt: Optional[str] = None
    enable_thinking: bool = False
    thinking_budget_tokens: Optional[int] = None
    latency: Optional[str] = None


@dataclass
class BedrockResponse:
    text: str
    latency_s: float
    raw_response: Dict[str, Any]


class BedrockClaudeClient:
    """Thin wrapper over the AWS Bedrock runtime API for Claude Sonnet."""

    def __init__(
        self,
        config: BedrockClientConfig | None = None,
        *,
        galileo_tracer: GalileoTracer | None = None,
    ) -> None:
        self.config = config or BedrockClientConfig()
        self._galileo = galileo_tracer
        try:
            self._client = boto3.client("bedrock-runtime", region_name=self.config.region)
        except (BotoCoreError, NoCredentialsError) as exc:  # pragma: no cover - network call
            raise RuntimeError(
                "Unable to initialize AWS Bedrock client. Ensure AWS credentials and "
                "region are configured via environment variables or ~/.aws config."
            ) from exc

    def invoke(
        self,
        prompt: str,
        params: BedrockInvocationParams | None = None,
        *,
        trace_metadata: Optional[Dict[str, Any]] = None,
        metadata_postprocessor: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
    ) -> BedrockResponse:
        params = params or BedrockInvocationParams()

        system_prompt = params.system_prompt or self.config.system_prompt
        system_blocks: List[Dict[str, str]] = [
            {"text": f"{system_prompt} Use {params.reasoning_effort} reasoning effort."}
        ]
        messages = [
            {
                "role": "user",
                "content": [
                    {"text": prompt},
                ],
            }
        ]

        max_tokens = params.max_tokens
        if params.enable_thinking and params.thinking_budget_tokens is not None:
            if max_tokens is not None and params.thinking_budget_tokens >= max_tokens:
                raise ValueError(
                    "max_tokens must be greater than thinking_budget_tokens when thinking is enabled"
                )

        inference_config = {
            "maxTokens": max_tokens,
        }
        # Bedrock Anthropic models reject setting temperature + topP together; only send whichever is provided.
        if params.temperature is not None:
            inference_config["temperature"] = params.temperature
        if params.top_p is not None and params.temperature is None:
            inference_config["topP"] = params.top_p

        additional_fields: Dict[str, Any] = {}
        if params.enable_thinking:
            thinking_payload: Dict[str, Any] = {"type": "enabled"}
            if params.thinking_budget_tokens is not None:
                thinking_payload["budget_tokens"] = params.thinking_budget_tokens
            additional_fields["thinking"] = thinking_payload

        kwargs: Dict[str, Any] = {
            "modelId": self.config.model_id,
            "messages": messages,
            "system": system_blocks,
            "inferenceConfig": inference_config,
        }

        if params.latency:
            kwargs["performanceConfig"] = {"latency": params.latency}

        if additional_fields:
            kwargs["additionalModelRequestFields"] = additional_fields

        trace_state = None
        if self._galileo and self._galileo.is_enabled:
            trace_state = self._galileo.start_trace(prompt=prompt, system_prompt=system_prompt)

        try:
            start = time.time()
            response = self._client.converse(**kwargs)
            latency = time.time() - start
        except (BotoCoreError, ClientError) as exc:  # pragma: no cover - network call
            if trace_state:
                self._galileo.log_failure(trace_state, exc, metadata=trace_metadata)
            raise RuntimeError(f"Bedrock invocation failed: {exc}") from exc

        output = response.get("output", {})
        message = output.get("message", {})
        content_blocks = message.get("content", [])
        fragments = [
            block.get("text", "")
            for block in content_blocks
            if isinstance(block, dict) and block.get("text")
        ]
        text = "".join(fragments).strip()

        combined_metadata: Dict[str, Any] = {}
        if trace_metadata:
            combined_metadata.update(trace_metadata)
        if metadata_postprocessor:
            try:
                extra_metadata = metadata_postprocessor(text)
            except Exception as exc:  # pragma: no cover - telemetry quality
                extra_metadata = {"metadata_postprocess_error": str(exc)}
            if extra_metadata:
                combined_metadata.update(extra_metadata)

        if trace_state:
            self._galileo.log_success(
                trace_state,
                model_id=self.config.model_id,
                response_text=text,
                usage=response.get("usage"),
                metadata=combined_metadata or None,
            )

        return BedrockResponse(text=text, latency_s=latency, raw_response=response)


__all__ = [
    "DEFAULT_MODEL_ID",
    "DEFAULT_REGION",
    "BedrockClientConfig",
    "BedrockInvocationParams",
    "BedrockResponse",
    "BedrockClaudeClient",
]
