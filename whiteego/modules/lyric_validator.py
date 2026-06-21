from __future__ import annotations

from typing import Any, Dict, List

from whiteego.modules.hu_syllable import (
    check_rhyme,
    count_syllables_line,
    get_last_word,
)


def validate_lyrics(
    text: str,
    expected_line_count: int = 4,
    rhyme_pairs: List[tuple[int, int]] | None = None,
) -> Dict[str, Any]:
    if rhyme_pairs is None:
        rhyme_pairs = [(1, 3)]  # 2. és 4. sor

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    line_data = []
    for idx, line in enumerate(lines):
        line_data.append(
            {
                "index": idx,
                "text": line,
                "syllables": count_syllables_line(line),
                "last_word": get_last_word(line),
            }
        )

    rhyme_results = []
    for a, b in rhyme_pairs:
        if a < len(lines) and b < len(lines):
            word_a = get_last_word(lines[a])
            word_b = get_last_word(lines[b])
            rhyme_results.append(
                {
                    "line_a": a,
                    "line_b": b,
                    "word_a": word_a,
                    "word_b": word_b,
                    "rhymes": check_rhyme(word_a, word_b, strict=False),
                }
            )

    return {
        "line_count": len(lines),
        "expected_line_count": expected_line_count,
        "line_count_ok": len(lines) == expected_line_count,
        "lines": line_data,
        "rhyme_results": rhyme_results,
        "all_rhymes_ok": all(item["rhymes"] for item in rhyme_results) if rhyme_results else True,
    }
