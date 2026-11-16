"""Tests for ClaudeAPIClient."""

from unittest.mock import patch

from claudecode.claude_api_client import ClaudeAPIClient


@patch("claudecode.claude_api_client.Anthropic")
def test_validate_api_access_uses_configured_model(mock_anthropic):
    client = ClaudeAPIClient(model="claude-sonnet-5-5", api_key="test-key")

    valid, error = client.validate_api_access()

    assert (valid, error) == (True, "")
    create = mock_anthropic.return_value.messages.create
    assert create.call_args.kwargs["model"] == "claude-sonnet-5-5"


@patch("claudecode.claude_api_client.Anthropic")
def test_validate_api_access_reports_api_error(mock_anthropic):
    mock_anthropic.return_value.messages.create.side_effect = Exception("not_found_error")
    client = ClaudeAPIClient(model="claude-sonnet-5-5", api_key="test-key")

    valid, error = client.validate_api_access()

    assert not valid
    assert "not_found_error" in error
