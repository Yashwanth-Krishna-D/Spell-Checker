"""
candidate_generator.py
----------------------
Generates a set of candidate correct words for a given input word x.

─── Strategy ────────────────────────────────────────────────────────────────
This is a **full vocabulary scan** approach — for every word in the dictionary
we compute the edit distance to x.

We scan the entire vocabulary loaded by the LanguageModel and collect every
word w such that levenshtein(x, w) ≤ MAX_EDIT_DISTANCE. (In this case 2)

Time  complexity : O(V × m × n)
    V = vocabulary size, m = len(x), n = average word length in vocabulary
    **n ~= 7 in the corpus we have taken**
Space complexity : O(V)  worst case (all words are candidates)

─── Non-word vs Real-word ───────────────────────────────────────────────────
• Non-word error (x not in vocabulary):
    The input word itself is not added (it's not a valid word).

• Real-word error (x is in vocabulary):
    The input word is included as a candidate (it might be correct), along
    with all vocabulary words within edit distance ≤ 2.  
    The channel model and language model together decide the best correction.
"""

from .language_model import LanguageModel
from .edit_distance import levenshtein

MAX_EDIT_DISTANCE: int = 2


class CandidateGenerator:
    def __init__(self, language_model: LanguageModel,
                 max_edit_distance: int = MAX_EDIT_DISTANCE):
        self.lm = language_model
        self.max_dist = max_edit_distance
        self._vocab: list = sorted(self.lm.vocabulary())

    def generate(self, word: str) -> list:        
        # Return a list of candidate corrections for *word*.

        word = word.lower()
        candidates = []
        word_len = len(word)

        # Length-based pruning
        for vocab_word in self._vocab:
            if abs(len(vocab_word) - word_len) > self.max_dist:
                continue
            dist = levenshtein(word, vocab_word)
            if dist <= self.max_dist:
                candidates.append((vocab_word, dist))


        if self.lm.in_vocabulary(word):
            # Add with distance 0 if not already present
            existing = {c[0] for c in candidates}
            if word not in existing:
                candidates.append((word, 0))

        # Ranking : priority 1 = edit distance, Priority 2 = word frequency
        candidates.sort(key=lambda pair: (pair[1], pair[0]))

        return candidates

    def is_real_word(self, word: str) -> bool:
        return self.lm.in_vocabulary(word.lower())