"""
language_model.py
-----------------
Unigram language model built from a word-frequency corpus file.

P(w) = count(w) / total_words

Laplace (add-1) smoothing is applied so that every candidate word —
even one that is absent from the corpus — still gets a non-zero probability,
preventing the noisy-channel score from collapsing to zero.

Smoothing formula:
    P_smooth(w) = (count(w) + 1) / (total_words + vocab_size)

Time  complexity (loading) : O(V)  where V = vocabulary size
Space complexity            : O(V)
Time  complexity (query)    : O(1) – dictionary lookup
"""

import os

class LanguageModel:
    def __init__(self, corpus_path: str):
        self.word_counts: dict = {}
        self.total_words: int = 0
        self.vocab_size: int = 0
        self._load_corpus(corpus_path)

    def _load_corpus(self, path: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Corpus file not found: {path}\n"
            )

        with open(path, encoding="utf-8") as fh:
            for line_num, raw_line in enumerate(fh, start=1):
                line = raw_line.strip()

                if not line or line.startswith("#"):
                    continue

                parts = line.split("\t")
                if len(parts) != 2:
                    continue

                word, count_str = parts
                word = word.lower()

                try:
                    count = int(count_str)
                except ValueError:
                    continue

                self.word_counts[word] = self.word_counts.get(word, 0) + count
                self.total_words += count

        self.vocab_size = len(self.word_counts)

    def prob(self, word: str) -> float:
        word = word.lower()
        count = self.word_counts.get(word, 0)
        return (count + 1) / (self.total_words + self.vocab_size)

    def count(self, word: str) -> int:
        return self.word_counts.get(word.lower(), 0)

    def in_vocabulary(self, word: str) -> bool:
        return word.lower() in self.word_counts

    def vocabulary(self) -> set:
        return set(self.word_counts.keys())