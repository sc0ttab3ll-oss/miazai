from __future__ import annotations

from typing import Dict, List

from whiteego.contracts.master_table import DERVISH_AMEN_MASTER_TABLE

import pyphen

HU_DIC = pyphen.Pyphen(lang="hu_HU")

MASTER_TABLE_TEMPLATES = {

    "verse_tension": {"scheme": "ABAB", "syllables": [8, 7, 8, 7]},

    "hook_v1": {"scheme": "ABBA", "syllables": [8, 6, 6, 8]},

    "hook_v2_mantra": {"scheme": "AAAA", "syllables": [4, 4, 4, 4]},

    "hook_v3": {"scheme": "AABB", "syllables": [8, 8, 7, 7]},

    "outro": {"scheme": "AABB", "syllables": [8, 8, 7, 7]},

    "bridge_child": {"free": True, "syllables": None},

    "bridge_adult": {"free": True, "syllables": None},

    "bridge_divine": {"free": True, "syllables": None},

}

def count_syllables_hu(word: str) -> int:

    cleaned = "".join(ch for ch in word.lower() if ch.isalpha())

    if not cleaned:

        return 0

    positions = HU_DIC.positions(cleaned)

    return len(positions) + 1

def count_syllables_line_hu(line: str) -> int:

    words = line.strip().split()

    return sum(count_syllables_hu(word) for word in words)

def validate_master_table_row(line: str, expected: int, tolerance: int = 0) -> bool:

    actual = count_syllables_line_hu(line)

    return abs(actual - expected) <= tolerance

def validate_stanza_against_template(

    lines: List[str],

    template_name: str,

    tolerance: int = 0,

) -> bool:

    template = get_template(template_name)

    if template.scheme == "FREE":
        return True

    expected_syllables = template.syllable_counts

    if len(lines) != len(expected_syllables):

        return False

    return all(

        validate_master_table_row(line, expected, tolerance=tolerance)

        for line, expected in zip(lines, expected_syllables)

    )

def get_template_scheme(template_name: str) -> str | None:

    return get_template(template_name).scheme

def count_syllables_word(word: str) -> int:

    return count_syllables_hu(word)

def count_syllables_line(line: str) -> int:

    return count_syllables_line_hu(line)

def validate_line(line: str, expected: int, tolerance: int = 0) -> tuple[bool, int]:

    actual = count_syllables_line(line)

    return abs(actual - expected) <= tolerance, actual

def get_last_word(line: str) -> str:

    words = line.strip().split()

    return words[-1].rstrip('.,!?;:') if words else ""

def get_rhyme_nucleus(word: str) -> str:

    vowels = "aáeéiíoóöőuúüű"

    word_lower = word.lower().rstrip('.,!?;:')

    last_vowel_pos = -1

    for i, ch in enumerate(word_lower):

        if ch in vowels:

            last_vowel_pos = i

    if last_vowel_pos == -1:

        return word_lower

    return word_lower[last_vowel_pos:]

def check_rhyme(word_a: str, word_b: str, strict: bool = False) -> bool:

    if strict:

        return get_rhyme_nucleus(word_a) == get_rhyme_nucleus(word_b)

    vowel_pairs = {

        'á': 'a',

        'é': 'e',

        'í': 'i',

        'ó': 'o',

        'ő': 'ö',

        'ú': 'u',

        'ű': 'ü',

    }

    def normalize(s: str) -> str:

        return ''.join(vowel_pairs.get(c, c) for c in s)

    return normalize(get_rhyme_nucleus(word_a)) == normalize(get_rhyme_nucleus(word_b))

def get_template(template_name: str):

    return DERVISH_AMEN_MASTER_TABLE[template_name]
