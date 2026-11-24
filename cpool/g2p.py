"""g2pk2 surface pronunciation (notebook step), patched for Windows.

Patches (behaviour unchanged): use python-mecab-ko instead of `eunjeon`, never
shell out to a global `pip install`, read idioms.txt once instead of on every
call (Windows Errno 22 after many re-opens).
"""
from __future__ import annotations

from functools import lru_cache

_G2P = None


def _g2p():
    global _G2P
    if _G2P is None:
        import re
        import mecab
        from g2pk2 import G2p

        class Patched(G2p):
            _rules = None

            def check_mecab(self):
                return None

            def get_mecab(self):
                return mecab.MeCab()

            def idioms(self, string, descriptive=False, verbose=False):
                if Patched._rules is None:
                    rules = []
                    with open(self.idioms_path, encoding="utf8") as f:
                        for line in f:
                            line = line.split("#")[0].strip()
                            if "===" in line:
                                rules.append(tuple(line.split("===")))
                    Patched._rules = rules
                for a, b in Patched._rules:
                    string = re.sub(a, b, string)
                return string

        _G2P = Patched()
    return _G2P


@lru_cache(maxsize=None)
def surface(word: str) -> str:
    return _g2p()(word)


def _batch(words):
    return [(w, surface(w)) for w in words]


def surfaces_parallel(words, workers=0) -> dict:
    """~0.15 s/word; spread over processes (call from a script's __main__)."""
    import os
    from concurrent.futures import ProcessPoolExecutor
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    chunks = [words[i::workers * 4] for i in range(workers * 4)]
    out = {}
    failed = []
    try:
        with ProcessPoolExecutor(workers) as ex:
            futs = {ex.submit(_batch, ch): i for i, ch in enumerate(chunks)}
            for f, i in futs.items():
                try:
                    out.update(dict(f.result()))
                except Exception as e:      # Windows transient file errors in workers
                    failed.append(i)
    except Exception:
        failed = [i for i, ch in enumerate(chunks) if not all(w in out for w in ch)]
    for i in failed:                         # deterministic: same function, in-process
        out.update(dict(_batch(chunks[i])))
    missing = [w for w in words if w not in out]
    assert not missing, f"{len(missing)} words without surface form"
    return out
