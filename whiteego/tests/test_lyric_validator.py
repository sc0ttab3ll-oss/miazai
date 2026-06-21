from whiteego.modules.lyric_validator import validate_lyrics


def test_validate_lyrics_detects_line_count_and_rhyme_failure():
    text = """Ragyog az eső, a múlt árnyai,
Szívekbe mélyed, s fájdalom halványul.
Új fényfelé, egy csillag emelkedik,
A remény halkan szól, elfeledtetve."""

    result = validate_lyrics(text)

    assert result["line_count"] == 4
    assert result["line_count_ok"] is True
    assert result["all_rhymes_ok"] is False

    assert result["lines"][0]["syllables"] == 8
    assert result["lines"][1]["syllables"] == 12
    assert result["lines"][2]["syllables"] == 10
    assert result["lines"][3]["syllables"] == 11

    rhyme = result["rhyme_results"][0]
    assert rhyme["word_a"] == "halványul"
    assert rhyme["word_b"] == "elfeledtetve"
    assert rhyme["rhymes"] is False
