# Wordle word-list data

Game vocabulary for the Wordle RL project. These files define the environment: which
words can be the hidden secret, and which guesses are legal.

## Files

| File | Count | Meaning |
|---|---|---|
| `answers.txt` | **2,315** | Secret-word pool. A game's answer is drawn only from here. |
| `allowed_guesses.txt` | **12,972** | Full legal-guess pool (answers ∪ non-answers). Any guess must be in here. |
| `raw/solutions.txt` | 2,315 | Original downloaded source for answers (JS-array format). |
| `raw/nonsolutions.txt` | 10,657 | Original downloaded non-answer guesses (JS-array format). |

`allowed_guesses.txt` = `solutions.txt` ∪ `nonsolutions.txt` → 2,315 + 10,657 = **12,972**.

## Provenance

- **Source:** [`LaurentLessard/wordlesolver`](https://github.com/LaurentLessard/wordlesolver), branch `main`.
  - `answers.txt`         ← `solutions.txt`
  - non-answer guesses    ← `nonsolutions.txt`
- **Retrieved:** 2026-06-04 via `curl` from `raw.githubusercontent.com`.
- **Original origin:** these are the **original, pre-NYT** Wordle lists, extracted from the
  game's source code.

## Why the original (pre-NYT) list

After the NYT acquired Wordle (Nov 2022), they stopped using the fixed 2,315-word answer
list and now hand-curate answers per day. There is therefore no living "official" answer
set to benchmark against. The **original 2,315-word list is the universally benchmarked
target** — every "≈99% / 100% solver" result in the literature refers to it. The repo also
ships `solutions_nyt.txt` / `nonsolutions_nyt.txt` variants; **we do not use them.**

## Normalization

Both canonical files are: lowercase, exactly 5 letters `a`–`z`, one word per line, sorted,
deduplicated. Verified at creation:
- `answers.txt` = 2,315 lines, `allowed_guesses.txt` = 12,972 lines
- every word matches `^[a-z]{5}$`
- `answers ⊆ allowed_guesses`
