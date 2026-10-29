"""Streamed chat completions for the background LLM operations.

Some gateways drop a request that sends nothing back for minutes, which is what a long
consolidation or mental-model refresh does when it is not streamed. With
``HINDSIGHT_API_{CONSOLIDATION,MENTAL_MODEL_REFRESH}_LLM_STREAM`` the OpenAI-compatible
provider sends ``stream: true`` and folds the chunks back into one completion, so every
caller reads the same result it would unstreamed.
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from openai import APIError
from openai.types.chat import ChatCompletionChunk
from pydantic import BaseModel

from hindsight_api import MemoryEngine
from hindsight_api.config import clear_config_cache, get_config
from hindsight_api.engine.llm_wrapper import LLMProvider
from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM

_REQUEST = httpx.Request("POST", "https://litellm.internal/v1/chat/completions")


class _Answer(BaseModel):
    answer: str


def _chunk(delta: dict[str, Any] | None = None, *, finish_reason: str | None = None, usage: Any = None):
    choices = [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
    return ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "m",
            "choices": choices,
            "usage": usage,
        }
    )


def _text_chunks(*parts: str) -> list[ChatCompletionChunk]:
    chunks = [_chunk({"role": "assistant", "content": ""})]
    chunks += [_chunk({"content": part}) for part in parts]
    chunks.append(_chunk({}, finish_reason="stop"))
    chunks.append(_chunk(usage={"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}))
    return chunks


class _FakeStream:
    """The slice of the SDK's ``AsyncStream`` the provider uses."""

    def __init__(self, chunks: list[ChatCompletionChunk], error: BaseException | None = None):
        self._chunks = chunks
        self._error = error
        self.response = httpx.Response(200, request=_REQUEST)
        self.closed = False

    async def __aenter__(self) -> "_FakeStream":
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.closed = True

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for chunk in self._chunks:
            yield chunk
        if self._error is not None:
            raise self._error


def _llm(stream: bool) -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        provider="openai", api_key="k", base_url="https://litellm.internal/v1", model="m", stream=stream
    )


async def _call(llm: OpenAICompatibleLLM, *responses: Any, **kwargs: Any) -> Any:
    llm._client.chat.completions.create = AsyncMock(side_effect=list(responses))
    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"):
        return await llm.call(messages=[{"role": "user", "content": "hi"}], initial_backoff=0, **kwargs)


@pytest.mark.asyncio
async def test_structured_call_is_streamed_and_reassembled() -> None:
    llm = _llm(stream=True)
    stream = _FakeStream(_text_chunks('{"answer"', ': "blue', '"}'))

    result = await _call(llm, stream, response_format=_Answer, max_retries=0)

    assert result.content == _Answer(answer="blue")
    assert result.usage.input_tokens == 11
    assert result.usage.output_tokens == 7
    kwargs = llm._client.chat.completions.create.call_args.kwargs
    assert kwargs["stream"] is True
    assert kwargs["stream_options"] == {"include_usage": True}
    assert stream.closed


@pytest.mark.asyncio
async def test_free_form_call_is_streamed_and_reassembled() -> None:
    result = await _call(_llm(stream=True), _FakeStream(_text_chunks("Hello", ", world")), max_retries=0)

    assert result.content == "Hello, world"


@pytest.mark.asyncio
async def test_tool_call_arguments_are_reassembled_across_chunks() -> None:
    llm = _llm(stream=True)
    tool_call = {"index": 0, "id": "call_1", "type": "function", "function": {"name": "recall", "arguments": ""}}
    chunks = [
        _chunk({"role": "assistant", "tool_calls": [tool_call]}),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"query": '}}]}),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"paris"}'}}]}),
        _chunk({}, finish_reason="tool_calls"),
    ]
    llm._client.chat.completions.create = AsyncMock(return_value=_FakeStream(chunks))

    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"):
        result = await llm.call_with_tools(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "recall", "parameters": {"type": "object"}}}],
            max_retries=0,
        )

    assert [(tc.id, tc.name, tc.arguments) for tc in result.tool_calls] == [("call_1", "recall", {"query": "paris"})]
    assert result.finish_reason == "tool_calls"
    assert llm._client.chat.completions.create.call_args.kwargs["stream"] is True


@pytest.mark.asyncio
async def test_unstreamed_by_default() -> None:
    llm = _llm(stream=False)
    response = SimpleNamespace(
        error=None,
        usage=None,
        choices=[
            SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="hi", tool_calls=None, refusal=None))
        ],
    )

    result = await _call(llm, response, max_retries=0)

    assert result.content == "hi"
    kwargs = llm._client.chat.completions.create.call_args.kwargs
    assert "stream" not in kwargs
    assert "stream_options" not in kwargs


@pytest.mark.asyncio
async def test_connection_lost_mid_stream_is_retried() -> None:
    """The SDK wraps transport errors only while sending; mid-stream ones must still retry."""
    llm = _llm(stream=True)
    dropped = _FakeStream(_text_chunks("par")[:2], error=httpx.RemoteProtocolError("peer closed connection"))

    result = await _call(llm, dropped, _FakeStream(_text_chunks("complete")), max_retries=1)

    assert result.content == "complete"
    assert llm._client.chat.completions.create.call_count == 2


@pytest.mark.asyncio
async def test_error_event_mid_stream_is_retried() -> None:
    """Unstreamed, an upstream failure is a 5xx, which is retried; streamed it is an error event."""
    llm = _llm(stream=True)
    failed = _FakeStream(_text_chunks("par")[:2], error=APIError("upstream overloaded", _REQUEST, body=None))

    result = await _call(llm, failed, _FakeStream(_text_chunks("complete")), max_retries=1)

    assert result.content == "complete"


@pytest.mark.asyncio
async def test_empty_stream_is_retried() -> None:
    llm = _llm(stream=True)

    result = await _call(llm, _FakeStream([]), _FakeStream(_text_chunks("complete")), max_retries=1)

    assert result.content == "complete"


def test_unsupported_provider_warns_instead_of_dropping_the_setting(caplog) -> None:
    with caplog.at_level("WARNING", logger="hindsight_api.engine.llm_wrapper"):
        llm = LLMProvider(provider="mock", api_key="", base_url="", model="m", stream=True)

    assert llm.stream is True
    assert "not supported for provider 'mock'" in caplog.text


# ── config + engine wiring ──────────────────────────────────────────────────────

BASE_ENV = {
    "HINDSIGHT_API_SKIP_LLM_VERIFICATION": "true",
    "HINDSIGHT_API_LLM_PROVIDER": "mock",
    "HINDSIGHT_API_LLM_MODEL": "default-model",
}


@pytest.fixture
def engine_env(monkeypatch):
    def _build(**extra: str) -> MemoryEngine:
        for key, value in {**BASE_ENV, **extra}.items():
            monkeypatch.setenv(key, value)
        clear_config_cache()
        return MemoryEngine(skip_llm_verification=True)

    yield _build
    clear_config_cache()


def test_unset_streams_nothing_and_keeps_refresh_on_the_reflect_config(engine_env) -> None:
    engine = engine_env(HINDSIGHT_API_LLM_TIMEOUT="30", HINDSIGHT_API_REFLECT_LLM_TIMEOUT="30")

    assert get_config().consolidation_llm_stream is None
    assert get_config().has_mental_model_refresh_llm_override() is False
    assert engine._consolidation_llm_config.stream is False
    assert engine._mental_model_refresh_llm_config is engine._reflect_llm_config


def test_consolidation_stream_reaches_only_consolidation(engine_env) -> None:
    engine = engine_env(HINDSIGHT_API_CONSOLIDATION_LLM_STREAM="true")

    assert engine._consolidation_llm_config.stream is True
    assert engine._retain_llm_config.stream is False
    assert engine._reflect_llm_config.stream is False
    assert engine._mental_model_refresh_llm_config.stream is False


def test_refresh_stream_gets_its_own_llm_and_leaves_reflect_unstreamed(engine_env) -> None:
    engine = engine_env(HINDSIGHT_API_MENTAL_MODEL_REFRESH_LLM_STREAM="true")

    assert engine._mental_model_refresh_llm_config.stream is True
    assert engine._mental_model_refresh_llm_config.model == "default-model"
    assert engine._reflect_llm_config.stream is False
    assert engine._consolidation_llm_config.stream is False


def test_stream_reaches_every_member_of_a_failover_chain(engine_env) -> None:
    engine = engine_env(
        HINDSIGHT_API_CONSOLIDATION_LLM_STREAM="true",
        HINDSIGHT_API_LLM_STRATEGY='{"mode": "failover"}',
        HINDSIGHT_API_LLM_1_PROVIDER="mock",
        HINDSIGHT_API_LLM_1_MODEL="backup-model",
    )

    members = engine._consolidation_llm_config.members
    assert [m.stream for m in members] == [True, True]


def test_invalid_value_fails_at_startup(monkeypatch) -> None:
    monkeypatch.setenv("HINDSIGHT_API_CONSOLIDATION_LLM_STREAM", "yes please")
    clear_config_cache()
    try:
        with pytest.raises(ValueError, match="HINDSIGHT_API_CONSOLIDATION_LLM_STREAM"):
            get_config()
    finally:
        clear_config_cache()
