import json
from unittest.mock import MagicMock

import pytest
import requests

from nusecast.assistant import AssistantError, Settings, ask, load_settings


@pytest.fixture
def transport(monkeypatch):
    post = MagicMock()
    response = post.return_value
    response.status_code = 200
    response.json.return_value = {"status": "completed", "output": [
        {"type": "message", "content": [{"type": "output_text", "text": "MAE de RN: 36,44."}]}]}
    monkeypatch.setattr("nusecast.assistant.requests.post", post)
    return post


def test_request_is_bounded_and_does_not_store(transport):
    settings = Settings("test-private-value", enabled=True)
    assert "36,44" in ask(settings, "¿Qué MAE?", {"mae": 36.44})
    assert settings.api_key not in repr(settings)
    args = transport.call_args.kwargs
    assert args["json"]["store"] is False
    assert args["json"]["max_output_tokens"] == 600
    assert args["allow_redirects"] is False
    assert "tools" not in args["json"]
    assert settings.api_key not in json.dumps(args["json"])


@pytest.mark.parametrize("status", [301, 400, 401, 403, 404, 429, 500])
def test_errors_never_echo_service_body(transport, status):
    transport.return_value.status_code = status
    transport.return_value.text = "private-value"
    with pytest.raises(AssistantError) as error:
        ask(Settings("private-value", enabled=True), "Pregunta", {})
    assert "private-value" not in str(error.value)
    transport.return_value.json.assert_not_called()
    assert transport.call_count == 1


def test_timeout_is_safe(transport):
    transport.side_effect = requests.Timeout("private-value")
    with pytest.raises(AssistantError, match="conectar"):
        ask(Settings("private-value", enabled=True), "Pregunta", {})


@pytest.mark.parametrize("question", ["", "x" * 2001, "clave sk-not-a-real-secret-123456"])
def test_invalid_question_never_sent(transport, question):
    with pytest.raises(AssistantError):
        ask(Settings("test", enabled=True), question, {})
    transport.assert_not_called()


def test_disabled_never_calls_network(transport):
    with pytest.raises(AssistantError):
        ask(Settings(), "Pregunta", {})
    transport.assert_not_called()


def test_malformed_response(transport):
    transport.return_value.json.return_value = {"output": []}
    with pytest.raises(AssistantError, match="utilizable"):
        ask(Settings("test", enabled=True), "Pregunta", {})


def test_settings_rotation_and_no_shell_evaluation(tmp_path, monkeypatch):
    for key in ["OPENAI_API_KEY", "OPENAI_ENABLED", "OPENAI_MODEL", "OPENAI_MAX_OUTPUT_TOKENS"]:
        monkeypatch.delenv(key, raising=False)
    assert not load_settings(tmp_path).ready
    config = tmp_path / ".env"
    config.write_text('OPENAI_API_KEY="fake-one"\nOPENAI_ENABLED=true\nOPENAI_MAX_OUTPUT_TOKENS=999999\n')
    assert load_settings(tmp_path).ready
    assert load_settings(tmp_path).max_tokens == 1000
    config.write_text('OPENAI_API_KEY=fake-two\nOPENAI_ENABLED=true\nOPENAI_MAX_OUTPUT_TOKENS=invalid\n')
    assert load_settings(tmp_path).api_key == "fake-two"
    assert load_settings(tmp_path).max_tokens == 600
    monkeypatch.setenv("OPENAI_API_KEY", "process-value")
    assert load_settings(tmp_path).api_key == "process-value"


def test_echoed_key_is_redacted(transport):
    transport.return_value.json.return_value["output"][0]["content"][0]["text"] = "private-value"
    assert ask(Settings("private-value", enabled=True), "Pregunta", {}) == "[clave ocultada]"
