# Wordle Model — Requirements

> Status: **Requirements agreed; pre-implementation.** Core requirements are settled
> (see §5 decisions log). Implementation has **not** started — the user has further
> questions to work through first. Items marked **OPEN** are unresolved; **DECIDED** are settled.

## 1. Goal (one line)

Build a system that plays the Wordle game. **Target: 100% win rate.**

## 2. Background: the game

Standard Wordle:
- A hidden secret word is chosen from an *answer set*.
- The player has **6 guesses**; each guess must be a valid 5-letter word.
- After each guess, per-letter feedback is returned:
  - 🟩 green = correct letter, correct position
  - 🟨 yellow = letter in the word, wrong position
  - ⬛ gray = letter not in the word (accounting for duplicates)
- "Win" = the secret word is guessed within 6 attempts.
- **100% win rate** = every word in the answer set is solved in ≤6 guesses.

## 3. Key decisions

### 3.1 What "model" means — DECIDED ✅
The model is a **tiny character-level transformer, trained from scratch** — a genuine
learned neural policy (option (b) below). We considered three architectures:

| Option | Description | Can it guarantee 100%? |
|---|---|---|
| (a) Algorithmic solver | Entropy / minimax search over candidates. Not "trained" in the ML sense. | **Yes** — a fixed strategy (e.g. opening `SALET`) provably solves all 2,315 standard answers in ≤6. |
| (b) Learned policy (RL/self-play) ← **chosen** | A real neural net trained via self-play. | Hard — RL agents typically plateau at ~98–99%. |
| (c) Hybrid | Train a model, but keep a verified search fallback so the guarantee holds. | **Yes**, via the fallback. |

**Resolved tension:** the hard requirement is that it is *genuinely a trained ML model*
— learning is the point. 100% is **aspirational**, not a hard guarantee; ~98–99% is
acceptable. (This is why pure (b) was chosen over (a)/(c).) See §3.6 for *how* it learns.

*Why char-level:* normal LLMs tokenize into subword chunks, so they cannot "see"
individual letters — exactly the reasoning Wordle needs. A char-level model treats each
character as its own token, making letter-level reasoning natural.

### 3.2 Rules variant — DECIDED ✅
**Standard Wordle:** 5 letters, 6 guesses, normal mode, duplicate-aware feedback.
No variants.

### 3.3 Word list — DECIDED ✅
**Official NYT list:** ~2,315 answer words + ~12,972 valid guess words. Answers are drawn
only from the ~2,315; guesses may be any valid word. (Note: the allowed-guess count varies
slightly by source — we will pick and document one source during implementation.)

### 3.4 Verification — DECIDED ✅
To claim 100%, we run the player against **every** word in the answer set and
assert zero failures. This eval harness is part of the deliverable.

### 3.5 Compute / runtime budget — DECIDED ✅
**Everything must run on the user's laptop** — a MacBook Pro with Apple **M4 Max**
chip and **36 GB** unified memory. No cloud GPUs, no clusters.

Implications:
- Training (if any) must fit in ~36 GB unified memory and complete in reasonable
  time on Apple Silicon. Favors Apple's **MPS** backend (PyTorch/MLX) or CPU; rules
  out large models / multi-GPU regimes.
- An **algorithmic solver (3.1a)** is trivially within budget (runs in seconds on CPU).
- An **RL agent (3.1b)** is feasible only if kept small (the Wordle state/action
  space is tiny), but must be sized to this machine.
- This makes **fine-tuning a large LLM** impractical locally and effectively off the
  table. A *tiny* char-level transformer (§3.1) fits trivially.

### 3.6 How the model learns — DECIDED ✅
**Reinforcement learning via self-play** — the model *discovers* strategy itself (not
behavioral cloning, not imitating a hand-coded solver). Planned algorithm: **PPO** with a
value head and GAE, written from scratch (minimal dependencies; learning is the goal).

**Action space (how a char-level model emits a valid guess)** — two designs behind one
shared interface, so the env / PPO / reward / eval stack is built once:
- **D1 (baseline, build first):** char-level *input* encoding of the game transcript +
  a **word-level policy head** = softmax over the valid-guess vocabulary. One guess = one
  clean categorical action, always valid. The reliable path to ~98–99%.
- **D2 (stretch, fully char-level):** char-by-char decoder generating 5 letters,
  **trie-masked** to valid words. Genuinely char-level end-to-end, but higher RL variance.
  Swapped in behind the same interface after D1 works.

Supporting techniques planned: a **curriculum** (small answer set → 2,315 answers → full
guess vocab), **potential-based reward shaping** (info-gain / candidate-set reduction,
annealed to a pure win reward), and an entropy bonus + KL early-stop for stability.

### 3.7 Efficiency as a goal — DECIDED ✅
Primary metric is **win rate**. Average guess count is a **secondary** goal: the reward is
speed-scaled (solving in fewer guesses is better), but we optimize win rate first.

### 3.8 Game start / opening guess — DECIDED ✅
**Turn 1 is a fixed strong opener (`salet`); the model plays turns 2–6.** Every game begins
by playing a fixed, strong first word; the environment returns its feedback, and the model
takes over from turn 2 with that context already on the board. Applies to **both training
and eval** (no train/eval mismatch).

Rationale:
- Turn 1 (empty context) is the only zero-information, highest-variance decision; removing
  it lets the model focus its capacity on the harder mid/endgame.
- The model never faces an empty board — it always reasons from real feedback.
- The opener must be **strong, not random**: a random opener from the 12,972 pool is usually
  weak (rare/repeated letters), which would cap win rate and/or create off-distribution
  states. A fixed strong opener (e.g. `salet`, the lowest-average opener on this answer set)
  keeps win rate high. This is also how many top solver strategies work.

Notes:
- The opener word is a **config value** (default `salet`); easy to change/ablate.
- Optional future refinement (not adopted now): sprinkle random openers into *training only*
  for extra exploration while keeping the fixed opener at eval ("exploring starts").
- This means the model's learned policy covers turns 2–6; turn 1 is not a learned decision.

### 3.9 Learning-history logging — DECIDED ✅
During training we record the **learning history in a generic, structured, model-agnostic
format** (JSONL) — **semantic values, never the tokenized sequence fed to the model**.
Four streams per run (folder `wm/runs/<run_id>/`), all keyed by `iter`:
- `config.json` — hyperparameters, seed, git commit, opener, data version (reproducibility).
- `metrics.jsonl` — per-PPO-update learning curve + RL diagnostics.
- `eval.jsonl` — per-full-sweep win rate, guess distribution, and failed words.
- `games.jsonl` — a *sampled* subset of semantic game transcripts (guesses, `BYG` feedback,
  rewards, candidate counts, per-turn behavior `flags`) for debugging.

> **M1 status:** built = `config.json` / `metrics.jsonl` / `eval.jsonl`. **`games.jsonl`
> sampling and the guess-behavior diagnostics below are planned, not yet implemented.**

Includes (planned) **guess-behavior diagnostics** (`repeat_rate`, `inconsistent_rate`,
`reused_gray_rate`, `ignored_green_rate`, `misplaced_yellow_rate`) — descriptive rates over
training, **not** penalties (see §3.10).

Feedback is logged as a `BYG` string (e.g. `"BBYBG"`), not token IDs. We log every iteration
metric/eval but only a sampled subset of full games (self-play volume is huge). Full schemas:
**see `docs/logging.md`.**

### 3.10 Reward function — DECIDED ✅
- **Normal mode / full freedom:** the model may guess any valid word every turn. **No**
  consistency/hard-mode masking and **no** consistency penalties (using a gray letter, moving
  a green, dropping a yellow, etc. are *allowed* — they enable probe guesses, which are often
  optimal). Forcing consistency = Hard Mode, which loses the probe tactic and likely caps win
  rate below 100%. *(Note: during the training **curriculum** the legal guess pool is
  restricted per stage and grown toward the full 12,972 — a training ramp, not hard-mode
  masking; see `docs/design.md` → Curriculum.)*
- **Winning dominates speed:** big reward for any win, tiny bonus for fewer guesses
  (`R_win = 10 + 0.5·(6−k)`, loss `= 0`). The model never gambles a win for speed.
- **Shaping ON ("warmer" hints):** potential-based candidate-set reduction
  (`Φ = −log₂|consistent answers|`) — **the key tunable.** Sized so a full winning game's
  warmer rewards ≈ **⅓–½ of a win** (default coef `0.8`), **never ≥ a win**, annealed to 0.
  Rewards smart narrowing/probes, nothing for wasteful repeats, provably non-distorting.
  (Default; pure terminal reward is the alternative.)
- **No penalties** (consistency/repeat/validity): the policy head only emits valid words, and
  a repeated guess is already self-punishing via the speed term + zero shaping.
- **Exact repeats: not masked.** Re-guessing an already-tried word is the one always-wasteful
  move, but we rely on the reward to teach the model to avoid it (it earns nothing + wastes a
  turn) rather than masking it out. We **track** it via the `repeat_rate` diagnostic (§3.9) to
  confirm it falls toward 0.
- **"Bad guess" behaviors → diagnostics, not rules.** The earlier list (repeats, reusing
  grays, ignoring greens, misplacing yellows, inconsistent guesses) is *logged* as descriptive
  metrics over training (see §3.9 / `docs/logging.md`), **not** penalized — the reward already
  judges each guess on its informativeness (good probes rewarded, wasteful guesses not).

Full spec and the `-OUND` probe rationale: **see `docs/design.md` → Reward function.**

## 4. Open questions (to resolve next)
_All requirements-level questions are currently resolved. The user has further questions
to work through before implementation begins; new items will be logged here._

## 5. Decisions log
- **Runtime budget (§3.5):** Must run entirely on a MacBook Pro M4 Max, 36 GB RAM.
  No cloud/cluster compute. (decided)
- **Model (§3.1):** Tiny **char-level transformer, trained from scratch** — a genuine
  learned policy. 100% aspirational, ~98–99% acceptable. (decided)
- **Rules (§3.2):** Standard Wordle — 5 letters, 6 guesses, duplicate-aware feedback. (decided)
- **Word list (§3.3):** Official NYT — ~2,315 answers + ~12,972 valid guesses. (decided)
- **Learning signal (§3.6):** RL self-play (PPO from scratch). Policy head: original **D1**
  free word-head → **replaced by a letter-grounded head** (M1 — the free head memorized;
  letter-grounding generalizes). **D2** char-decoder still a parked stretch. (updated M1)
- **Efficiency (§3.7):** Win rate primary; avg guess count secondary (speed-scaled reward). (decided)
- **Opening guess (§3.8):** Fixed strong opener `salet` on turn 1 (config-driven); model
  plays turns 2–6, in both training and eval. (decided)
- **Logging (§3.9):** Structured JSONL learning history (semantic, not tokenized) —
  `config.json` / `metrics.jsonl` / `eval.jsonl` / `games.jsonl` per run. See
  `docs/logging.md`. (decided)
- **Input format:** Interleaved one-token-per-cell — each cell = `LetterEmb + FeedbackEmb +
  TurnEmb + ColEmb` (summed); `CLS` pooling token; no `STEP`/`SEP` separators; max ~26
  tokens. Notation `(sB aY lB eY tB)(bB rG aG cY eG)`. See `docs/design.md`. (decided)
- **Reward / action space (§3.10):** Normal mode (full 12,972-word freedom, no masking, no
  consistency penalties); win-dominant reward `10 + 0.5·(6−k)`, loss `0`; potential-based
  candidate-reduction shaping (**key tunable**, default coef `0.8` ≈ ⅓–½ of a win, never ≥ a
  win, annealed to 0). See `docs/design.md`. (decided)
- **Bad-guess behaviors (§3.10/§3.9):** not penalized/masked — exact repeats handled by the
  reward; all behaviors (repeat/inconsistent/reused-gray/ignored-green/misplaced-yellow)
  *logged* as diagnostics over training. (decided)
- **Build scope — milestone ladder:** M0 (game scaffold + `play`, no ML) → M1 (prove it
  learns) → M2 (full-answer results) → M3 stretch (full 12,972 vocab + D2). Building **M0
  first**. See `docs/design.md` → M0. (decided)
- **`play` runner:** human-facing demo named `play` — default **100 games** (configurable
  `--n`), **rich live dashboard** + colored summary report; `evaluate()` is the quiet batched
  stats engine underneath. (decided)
- **Env / tooling:** `wm/` is its own git repo, managed with **uv** (`pyproject.toml` +
  `uv.lock`; Python 3.12; numpy, rich, pytest, torch). Run via `uv run`. (done)
- **Verification (§3.4):** Full sweep over all 2,315 answers; report win rate + guess
  distribution. (decided)
- **Word-list source:** Option B — `LaurentLessard/wordlesolver` (`solutions.txt` +
  `nonsolutions.txt`); vendored to `wm/data/` (2,315 answers / 12,972 guesses). (done)
- **M1 findings (OPEN):** pipeline learns on small sets (Stage A 50→87%) but the free D1 head
  **does not generalize** to the full answer set (it memorizes; held-out 76% seen / 2%
  unseen). Switched to a **letter-grounded head**; the decisive long full-set run is still
  pending. `rl/diagnostics.py` + `games.jsonl` sampling **deferred / not built**. See
  `docs/design.md` → Policy head / M1 status. (open)

## 6. Reference: detailed design
- **`docs/design.md`** — technical design: model architecture (**letter-grounded head**),
  input format, reward, model init, **curriculum & M1 status**.
- **`docs/logging.md`** — logging schemas (with built-vs-planned status).
- **`docs/results.md`** — experiments log: each training setting, its win-rate ceiling, and
  constraints.
