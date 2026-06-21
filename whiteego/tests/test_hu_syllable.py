from whiteego.modules.hu_syllable import (

    MASTER_TABLE_TEMPLATES,

    count_syllables_hu,

    count_syllables_line_hu,

    get_template_scheme,

    validate_master_table_row,

    validate_stanza_against_template,

)

def test_count_syllables_hu_simple_word():

    assert count_syllables_hu("alma") == 2

def test_count_syllables_hu_agglutinative_word():

    result = count_syllables_hu("megvalósíthatóságáról")

    assert result >= 6

def test_count_syllables_line_hu():

    line = "alma körte barack"

    assert count_syllables_line_hu(line) >= 5

def test_validate_master_table_row_exact_match():

    line = "almafa alatt ülök"

    actual = count_syllables_line_hu(line)

    assert validate_master_table_row(line, actual)

def test_validate_master_table_row_with_tolerance():

    line = "fényből épül a csend"

    actual = count_syllables_line_hu(line)

    assert validate_master_table_row(line, actual + 1, tolerance=1)

def test_validate_stanza_against_template_passes_for_free_template():

    lines = ["szabad sor", "másik szabad sor"]

    assert validate_stanza_against_template(lines, "bridge_child")

def test_validate_stanza_against_template_fails_on_wrong_line_count():

    lines = ["egy sor", "két sor"]

    assert not validate_stanza_against_template(lines, "hook_v1")

def test_get_template_scheme():

    assert get_template_scheme("hook_v1") == "ABBA"

def test_master_table_contains_expected_templates():

    assert "verse_tension" in MASTER_TABLE_TEMPLATES

    assert "hook_v2_mantra" in MASTER_TABLE_TEMPLATES

    assert "bridge_divine" in MASTER_TABLE_TEMPLATES