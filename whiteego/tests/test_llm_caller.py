from unittest.mock import Mock, patch

from requests.exceptions import ReadTimeout

from whiteego.modules.llm_caller import (
    LYRIC_MODEL,
    build_prompt,
    generate_lyrics,
    generate_psycho_reflection,
)


def test_build_prompt_includes_psycho_context_fields():
    psycho_context = {
        "dominant_emotion": "szomorúság",
        "intensity": "közepes",
        "polarity": "negatívból pozitívba forduló",
        "narrative_frame": "belső gyógyulás",
    }

    prompt = build_prompt(psycho_context)

    assert "szomorúság" in prompt
    assert "közepes" in prompt
    assert "negatívból pozitívba forduló" in prompt
    assert "belső gyógyulás" in prompt
    assert "csak a 4 sort add vissza" in prompt


@patch("whiteego.modules.llm_caller.requests.post")
def test_generate_lyrics_uses_default_model_and_returns_response(mock_post):
    mock_response = Mock()
    mock_response.json.return_value = {"response": "első sor\nmásodik sor"}
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    psycho_context = {
        "dominant_emotion": "öröm",
        "intensity": "enyhe",
        "polarity": "pozitív",
        "narrative_frame": "újrakezdés",
    }

    result = generate_lyrics(psycho_context)

    assert result == "első sor\nmásodik sor"
    mock_post.assert_called_once()

    _, kwargs = mock_post.call_args
    assert kwargs["json"]["model"] == LYRIC_MODEL
    assert kwargs["json"]["stream"] is False
    assert kwargs["timeout"] == 300


@patch("whiteego.modules.llm_caller.requests.post")
def test_generate_psycho_reflection_returns_fallback_on_timeout(mock_post):
    mock_post.side_effect = ReadTimeout()

    psycho_context = {
        "dominant_emotion": "sadness",
        "intensity": "medium",
        "polarity": "mixed",
        "narrative_frame": "inner healing",
    }

    result = generate_psycho_reflection(psycho_context)

    assert result == "Qwen reflection is currently unavailable within the allowed time on this machine."
