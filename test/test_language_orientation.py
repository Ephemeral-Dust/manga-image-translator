"""Tests for language tag normalization and render-direction inference.

Regression guard for the vertical-English disaster: clients (e.g. browser
extensions) legitimately send BCP-47 tags like ``en``; the orientation preset
only knew canonical codes like ``ENG``, so direction silently fell back to box
aspect ratio and English got typeset one glyph per line in tall balloons.
"""
import numpy as np

from manga_translator.utils.textblock import (
    TextBlock,
    canonical_language_code,
    resolve_language_orientation,
)


def test_canonical_language_code_aliases():
    assert canonical_language_code('en') == 'ENG'
    assert canonical_language_code('EN') == 'ENG'
    assert canonical_language_code('en-US') == 'ENG'
    assert canonical_language_code('eng') == 'ENG'
    assert canonical_language_code('ENG') == 'ENG'
    assert canonical_language_code('es') == 'ESP'
    assert canonical_language_code('spa') == 'ESP'
    assert canonical_language_code('ar') == 'ARA'
    assert canonical_language_code('ja') == 'JPN'
    assert canonical_language_code('jp') == 'JPN'
    assert canonical_language_code('pt-BR') == 'PTB'
    assert canonical_language_code('zh') == 'CHS'
    assert canonical_language_code('zh_CN') == 'CHS'
    assert canonical_language_code('zh-Hans') == 'CHS'
    assert canonical_language_code('zh-Hant') == 'CHT'
    assert canonical_language_code('zh-Hant-TW') == 'CHT'
    assert canonical_language_code('zh-TW') == 'CHT'
    assert canonical_language_code('zh-HK') == 'CHT'
    assert canonical_language_code(None) is None
    assert canonical_language_code('') is None
    # Unknown tags pass through upper-cased so preset lookups simply miss.
    assert canonical_language_code('klingon') == 'KLINGON'


def test_resolve_language_orientation():
    assert resolve_language_orientation('en') == 'h'
    assert resolve_language_orientation('en-US') == 'h'
    assert resolve_language_orientation('ENG') == 'h'
    assert resolve_language_orientation('ar') == 'hr'
    # CJK has no fixed orientation: report the default so callers use geometry.
    assert resolve_language_orientation('ja') == 'auto'
    assert resolve_language_orientation('zh-Hans') == 'auto'
    assert resolve_language_orientation('unknown') == 'auto'
    assert resolve_language_orientation(None) == 'auto'
    assert resolve_language_orientation('ja', default='h') == 'h'


def _block(target_lang, width, height, **kwargs):
    quad = np.array([[0, 0], [width, 0], [width, height], [0, height]])
    return TextBlock([quad], texts=['hello world'], target_lang=target_lang,
                     **kwargs)


def test_bcp47_target_lang_forces_horizontal_in_tall_box():
    # A tall manga balloon with an English target must render horizontally.
    assert _block('en', 40, 300).direction == 'h'
    assert _block('ENG', 40, 300).direction == 'h'


def test_japanese_target_still_falls_back_to_aspect_ratio():
    assert _block('ja', 40, 300).direction == 'v'
    assert _block('ja', 300, 40).direction == 'h'


def test_arabic_target_is_right_to_left():
    assert _block('ar', 300, 40).direction == 'hr'


def test_explicit_direction_override_beats_language_inference():
    assert _block('en', 40, 300, direction='v').direction == 'v'


def test_unknown_target_lang_falls_back_to_aspect_ratio():
    assert _block('klingon', 40, 300).direction == 'v'
