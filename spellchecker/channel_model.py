"""
channel_model.py
----------------
Implements the **channel (error) model** — the probability P(x | w):
"Given the intended word w, what is the probability the user typed x?"

─── Approach ────────────────────────────────────────────────────────────────
    P(x | w)  ∝  1 / (edit_distance(x, w) + 1)

Properties:
  • When x == w  →  edit distance = 0  →  P = 1.0   (best possible)
  • When dist = 1 →  P = 0.5
  • When dist = 2 →  P = 0.333...
  • Scores decay as distance grows, matching the intuition that closer
    candidates are more likely to be the intended word.

This is then used in Bayes' rule:
    score(w) = P(x | w) × P(w)

─── Handling Real-word errors ────────────────────────────────────────────────────────
When the input word x is itself a valid vocabulary word, the edit distance
x→x is 0, giving P(x|x) = 1.0.  The language model P(w) then breaks the
tie:  a more frequent near-miss word can outscore the input word, correctly
flagging a real-word error.
"""

from .edit_distance import levenshtein

class ChannelModel:
    def __init__(self):
        self._cache: dict = {}

    def prob(self, x: str, w: str) -> float:
        import math
        x, w = x.lower(), w.lower()
        if x == w:
            return 1.0
        
        key = (x, w)
        if key not in self._cache:
            dist = levenshtein(x, w)
            DECAY = 1.5
            self._cache[key] = math.exp(-dist * DECAY)

        return self._cache[key]

    def clear_cache(self) -> None:
        self._cache.clear()