"""Galileo tracing utilities for AWS Bedrock invocations."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

try:  # pragma: no cover - optional dependency
    from galileo import galileo_context
    from galileo.config import GalileoPythonConfig
except ImportError:  # pragma: no cover - optional dependency
    galileo_context = None  # type: ignore
    GalileoPythonConfig = None  # type: ignore


@dataclass
class GalileoTraceConfig:
    """Configuration for wiring Galileo tracing into Bedrock client calls."""

    enabled: bool = False
    project: Optional[str] = None
    log_stream: Optional[str] = None
    trace_name: str = "Bedrock Invocation"
    auto_load_dotenv: bool = True


@dataclass
class _TraceState:
    messages: List[Dict[str, str]]
    start_ns: int


class GalileoTracer:
    """Lightweight helper that mirrors the sample Galileo logging workflow."""

    def __init__(self, config: GalileoTraceConfig | None = None) -> None:
        self.config = config or GalileoTraceConfig()
        self._logger = None

        if not self.config.enabled:
            return

        if galileo_context is None:
            raise RuntimeError(
                "Galileo SDK is not installed. Install the 'galileo' package or disable tracing."
            )

        if self.config.auto_load_dotenv:
            # Allow credentials + console overrides to be sourced from .env
            load_dotenv()

        project = self.config.project or os.environ.get("GALILEO_PROJECT") or "BedrockClaude"
        log_stream = self.config.log_stream or os.environ.get("GALILEO_LOG_STREAM") or "bedrock-runs"

        galileo_context.init(project=project, log_stream=log_stream)
        self._logger = galileo_context.get_logger_instance()
        self._logger.start_session()

    @property
    def is_enabled(self) -> bool:
        return self._logger is not None

    def start_trace(self, prompt: str, system_prompt: str) -> Optional[_TraceState]:
        if not self._logger:
            return None

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        self._logger.start_trace(name=self.config.trace_name, input=prompt)
        return _TraceState(messages=messages, start_ns=time.time_ns())

    def log_success(
        self,
        state: _TraceState,
        *,
        model_id: str,
        response_text: str,
        usage: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._logger:
            return

        input_tokens, output_tokens, total_tokens = self._parse_usage(usage)
        duration_ns = time.time_ns() - state.start_ns
        self._logger.add_llm_span(
            input=state.messages,
            output=response_text,
            model=model_id,
            num_input_tokens=input_tokens,
            num_output_tokens=output_tokens,
            total_tokens=total_tokens,
            duration_ns=duration_ns,
        )
        self._logger.conclude(output=response_text)

    def log_failure(self, state: _TraceState, error: Exception) -> None:
        if not self._logger:
            return

        duration_ns = time.time_ns() - state.start_ns
        self._logger.add_llm_span(
            input=state.messages,
            output=f"Invocation failed: {error}",
            model="bedrock-error",
            num_input_tokens=None,
            num_output_tokens=None,
            total_tokens=None,
            duration_ns=duration_ns,
        )
        self._logger.conclude(output=str(error))

    def flush(self) -> None:
        if self._logger:
            self._logger.flush()

    def get_console_links(self) -> Optional[Dict[str, str]]:
        if not self._logger or GalileoPythonConfig is None:
            return None
        config = GalileoPythonConfig.get()
        project_url = f"{config.console_url}project/{self._logger.project_id}"
        log_stream_url = f"{project_url}/log-streams/{self._logger.log_stream_id}"
        return {"project": project_url, "log_stream": log_stream_url}

    @staticmethod
    def _parse_usage(usage: Optional[Dict[str, Any]]) -> tuple[Optional[int], Optional[int], Optional[int]]:
        if not usage:
            return None, None, None
        # Handle either snake_case (invoke_model) or camelCase (converse) payloads.
        input_tokens = usage.get("inputTokens") or usage.get("input_tokens")
        output_tokens = usage.get("outputTokens") or usage.get("output_tokens")
        total_tokens = usage.get("totalTokens") or usage.get("total_tokens")
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        return input_tokens, output_tokens, total_tokens


__all__ = ["GalileoTracer", "GalileoTraceConfig"]
