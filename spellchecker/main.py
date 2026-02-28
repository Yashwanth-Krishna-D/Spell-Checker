import argparse
import os
import sys
import time

current_path = os.path.dirname(os.path.abspath(__file__))
corpus_path = os.path.join(current_path, "corpus", "word_frequencies.txt")

lang_model = None
cand_generate = None
chan_model = None


def init_models():

    global lang_model, cand_generate, chan_model

    if lang_model is not None:
        return  

    from .language_model import LanguageModel
    from .candidate_generator import CandidateGenerator
    from .channel_model import ChannelModel
    
    print(f"\n...[Entered CLI Mode]...")
    print("\t>>Loading corpus ...", end="", flush=True)
    t0 = time.time()

    lang_model = LanguageModel(corpus_path)
    cand_generate = CandidateGenerator(lang_model)
    chan_model = ChannelModel()

    elapsed = time.time() - t0
    print(f"\r\t>>[OK] Corpus loaded: {lang_model.vocab_size:,} words "
          f"({elapsed:.2f}s)          ")

def spell_check(word: str, top_n: int = 10, verbose: bool = False) -> dict:
    init_models()
    word = word.strip().lower()

    # Step 1: Generate candidates
    raw_candidates = cand_generate.generate(word)  
    is_real_word = cand_generate.is_real_word(word)

    if not raw_candidates:
        return {
            "input": word,
            "is_real_word": is_real_word,
            "candidates": [],
            "best": None
        }

    # Step 2 & 3: Score and rank 
    scored = []
    for (candidate, dist) in raw_candidates:
        p_x_given_w = chan_model.prob(word, candidate)   # P(x|w)
        p_w = lang_model.prob(candidate)                 # P(w)
        score = p_x_given_w * p_w                        # Bayes product
        scored.append({
            "word": candidate,
            "edit_dist": dist,
            "channel_prob": p_x_given_w,
            "lang_prob": p_w,
            "score": score
        })

    # Sort by score descending; break ties alphabetically
    scored.sort(key=lambda d: (-d["score"], d["word"]))
    best = scored[0]["word"] if scored else None

    return {
        "input": word,
        "is_real_word": is_real_word,
        "candidates": scored[:top_n],
        "best": best
    }

def show(result: dict, verbose: bool = False) -> None:
    word = result["input"]
    is_rw = result["is_real_word"]
    candidates = result["candidates"]
    best = result["best"]

    print()
    print("\t>> " + "=" * 52)
    print(f"\t>>    Input  : {word}")
    word_type = "real-word (valid but possibly wrong)" if is_rw else "non-word (not in vocabulary)"
    print(f"\t>>    Type   : {word_type}")
    print("\t>> " + "=" * 52)

    if not candidates:
        print("\n  [!] No candidates found within edit distance 2.")
        print("  Try checking the spelling more carefully.\n")
        return
    
    print("\n\t>>   Candidates:")
    print(f"\t>>  {' #':<4} {'Word':<20} {'EditDist':<10}", end="")
    if verbose:
        print(f"\t\t{'P(x|w)':<12} {'P(w)':<14} {'Score':<16}", end="")
    print()
    print("\t" + "-" * (38 + (42 if verbose else 0)))

    for rank, cand in enumerate(candidates, start=1):
        line = f"  {rank:<4} {cand['word']:<20} {cand['edit_dist']:<10}"
        if verbose:
            line += (f" {cand['channel_prob']:<12.6f}"
                     f" {cand['lang_prob']:<14.8f}"
                     f" {cand['score']:<16.10f}")
        print(f'\t>> {line}')

    print()
    print(f"\t>>Best correction: {best}")
    print()

def operations() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m spellchecker.main",
        description=(
            "Python Spell Checker — Noisy Channel Model + Levenshtein Distance\n"
            "Supports both non-word errors and real-word errors."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m spellchecker.main\n"
            "  python -m spellchecker.main --word acress\n"
            "  python -m spellchecker.main --word acress --top 5 --verbose\n"
        )
    )
    parser.add_argument(
        "--word", "-w",
        type=str,
        default=None,
        help="Word to spell-check"
    )
    parser.add_argument(
        "--top", "-t",
        type=int,
        default=10,
        metavar="N",
        help="Number of top candidates to display"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed scoring (P(x|w), P(w), score)"
    )
    return parser


def get_input(top_n: int, verbose: bool) -> None:
    init_models()
    print("\n\t>>[SPELL CHECKER] Interactive Mode")
    print("\t>>Press Enter on an empty line or Ctrl+C to quit.\n")

    while True:
        try:
            word = input("\n\n\t>>Enter word: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n...End...")
            break

        if not word:
            print("\n...End...")
            break

        result = spell_check(word, top_n=top_n, verbose=verbose)
        show(result, verbose=verbose)


def main():
    parser = operations()
    args = parser.parse_args()
    if args.word:
        result = spell_check(args.word, top_n=args.top, verbose=args.verbose)
        show(result, verbose=args.verbose)
    else:
        get_input(top_n=args.top, verbose=args.verbose)

if __name__ == "__main__":
    main()