"""Hangul syllable algebra: syllable = 0xAC00 + 588*initial + 28*medial + final."""
from __future__ import annotations

import re

CHOSEONG = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
JUNGSEONG = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
JONGSEONG = [""] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")

HANGUL_WORD = re.compile(r"^[가-힣]+$")


def decompose(ch: str):
    """-> (initial, medial, final) or None for a non-syllable character."""
    code = ord(ch) - 0xAC00
    if code < 0 or code > 11171:
        return None
    return CHOSEONG[code // 588], JUNGSEONG[(code % 588) // 28], JONGSEONG[code % 28]


def is_hangul_word(w: str) -> bool:
    return bool(HANGUL_WORD.match(w))


def coda(ch: str) -> str:
    d = decompose(ch)
    return d[2] if d else ""


def onset(ch: str) -> str:
    d = decompose(ch)
    return d[0] if d else ""
