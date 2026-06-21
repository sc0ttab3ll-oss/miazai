
from dataclasses import dataclass

from typing import List, Literal

@dataclass

class StanzaTemplate:

    name: str

    scheme: Literal["ABAB", "ABBA", "AABB", "AAAA", "FREE"]

    syllable_counts: List[int]

    syllable_tolerance: int = 0

    notes: str = ""

DERVISH_AMEN_MASTER_TABLE = {

    "verse_tension": StanzaTemplate(

        name="verse_tension",

        scheme="ABAB",

        syllable_counts=[8, 7, 8, 7],

        notes="Tension sections. A-lines drive, B-lines resolve.",

    ),

    "hook_v1": StanzaTemplate(

        name="hook_v1",

        scheme="ABBA",

        syllable_counts=[8, 6, 6, 8],

        notes="Hook verse 1 — enclosed rhyme gives containment/resolution feel.",

    ),

    "hook_v2_mantra": StanzaTemplate(

        name="hook_v2_mantra",

        scheme="AAAA",

        syllable_counts=[4, 4, 4, 4],

        notes="Mantra mode. Short, repetitive.",

    ),

    "hook_v3": StanzaTemplate(

        name="hook_v3",

        scheme="AABB",

        syllable_counts=[8, 8, 7, 7],

    ),

    "outro": StanzaTemplate(

        name="outro",

        scheme="AABB",

        syllable_counts=[8, 8, 7, 7],

    ),

    "bridge_child": StanzaTemplate(

        name="bridge_child",

        scheme="FREE",

        syllable_counts=[],

        notes="Triple-Voice Bridge — Child voice. Free form.",

    ),

    "bridge_adult": StanzaTemplate(

        name="bridge_adult",

        scheme="FREE",

        syllable_counts=[],

        notes="Triple-Voice Bridge — Adult voice. Free form.",

    ),

    "bridge_divine": StanzaTemplate(

        name="bridge_divine",

        scheme="FREE",

        syllable_counts=[],

        notes="Triple-Voice Bridge — Divine voice. Free form.",

    ),

}

