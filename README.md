## Algorithm

### Core Idea — Noisy Channel Model
The spell checker treats user input as a "noisy" signal: the user intended to type word **w**, but the channel (keyboard, fingers, memory) introduced noise, producing observed string **x**.

Using **Bayes' Rule**, we find the most probable intended word:

```
argmax_w  P(w | x)  ∝  argmax_w  P(x | w) × P(w)
```

| Term | Name | Meaning |
|------|------|---------|
| `P(w)` | Language Model | How common is word `w` in English text? |
| `P(x\|w)` | Channel Model | If `w` was intended, how likely is typo `x`? |
| `P(w\|x)` | Posterior | Given observation `x`, how likely is `w`? |

---

### Levenshtein (Edit) Distance

**Custom cost table** as specified:

| Operation | Cost |
|-----------|------|
| Insertion | 1 |
| Deletion | 1 |
| Substitution (same character) | 0 |
| Substitution (diff character) | 2 |

Standard **dynamic programming** approach:

```
dp[i][j] = min(
    dp[i-1][j]   + 1,               # deletion
    dp[i][j-1]   + 1,               # insertion
    dp[i-1][j-1] + sub_cost(i, j)   # substitution
)
```

**Time complexity:** O(m × n) — m and n are lengths of the two strings.

---

### Language Model — P(w)

Unigram model with Laplace (add-1) smoothing to avoid zero probabilities:
```
P(w) = (count(w) + 1) / (total_tokens + vocab_size)
```

---

### Channel Model — P(x | w)

Inverse edit-distance:
```
P(x | w) = 1 / (edit_distance(x, w) + 1)
```
---

### Candidate Generation

1. All vocabulary words with `levenshtein(x, w) ≤ 2` are collected.
2. A **length-based filter**  assuming any word whose length differs from `x` by more than 2 is not a candidate.

---

## Project Structure

```
Assignment 1/
├── spellchecker/
│   ├── __init__.py              
│   ├── edit_distance.py         
│   ├── candidate_generator.py   
│   ├── language_model.py        
│   ├── channel_model.py         
│   ├── main.py                  
│   └── corpus/
│       └── word_frequencies.txt 
└── README.md
|__ .gitignore
|__ requirements.txt
|__ templates/
    |__ index.html
|__ app.py
|__ procfile
|__ runtime.txt
```

---
## How to Run

### Interactive Mode (REPL)

```bash
cd "Assignment 1"
python app.py
```