# Design (technical)

Technical design notes for the Wordle RL project. Requirements/decisions live in
`requirements.md`; logging schemas in `logging.md`. This doc holds implementation-level
design. It will grow as we lock details (env, PPO, reward); for now it covers the **model**.

## Model architecture (summary)

A tiny **character-level transformer** with two heads, trained from scratch via PPO
self-play. Char-level *input* encoding (the model reads each guess letter and its feedback);
the policy head differs by variant (see `requirements.md` §3.6):

- **Trunk:** pre-LN transformer. Default `d_model=256, n_layers=4, n_heads=4, d_ff=1024,
  dropout=0`. ~2–3M params.
- **Input / embeddings:** interleaved one-token-per-cell format — each board cell fuses
  four summed `d_model` embeddings (`LetterEmb + FeedbackEmb + TurnEmb + ColEmb`). Max
  sequence ≈ **26 tokens**. See **Input format** below.
- **Policy head:**
  - **D1 (baseline):** `h → logits[12,972]` over valid guesses (Categorical). ~3.3M params.
  - **D2 (stretch):** char-by-char decoder, trie-masked to valid words.
- **Value head:** `h → scalar` (used by PPO for advantages; ignored at eval).

Total D1 ≈ **~6.5M params**.

## Input format (interleaved cells)

The model reads the game board as a flat sequence of tokens, **one token per filled cell**.
The board is a grid: up to 6 turns × 5 columns. Each cell carries a letter and its color
(feedback). We fuse those — plus the cell's turn and column — into a single token vector.

### One cell = one token (four summed channels)
```
cell "rG" (letter r, green) at turn 2, column 2:

token = LetterEmb[r] + FeedbackEmb[G] + TurnEmb[2] + ColEmb[2]
        └─ 256-d ─────── 256-d ──────── 256-d ─────── 256-d ┘
                       elementwise sum  →  one 256-d vector
```
Summed (not concatenated) → the token stays `d_model`-sized. No dimension growth.

### Notation
Primary (compact transcript): each turn in parens, each cell as `letter`+`COLOR`:
```
(sB aY lB eY tB)(bB rG aG cY eG)
 └─ turn 1 ────┘ └─ turn 2 ────┘     B=gray  Y=yellow  G=green
```
**The parens and spaces are page-only notation — NOT tokens.** Turn separation is carried by
`TurnEmb` (turns are fixed 5-cell width), not by any separator token.

Two-row grid (for the channel breakdown):
```
turn:      —    1  1  1  1  1    2  2  2  2  2
letter:   CLS   s  a  l  e  t    b  r  a  c  e
feedback:  —    B  Y  B  Y  B    B  G  G  Y  G
column:    —    1  2  3  4  5    1  2  3  4  5
```

### Worked example
Secret `crane` (hidden). Fixed opener `salet`, then the model plays:
```
turn 1  salet → B Y B Y B   (fixed opener)
turn 2  brace → B G G Y G   (model)
turn 3  crane → G G G G G ✅ (model — win)
```
At the **turn-3 decision** the model sees the two completed turns — 11 tokens:
```
pos:    0     1    2    3    4    5    6    7    8    9    10
tok:  [CLS] (sB   aY   lB   eY   tB)  (bB   rG   aG   cY   eG)
            └──── turn 1 ────┘        └──── turn 2 ────┘
```
At the earlier **turn-2 decision** it sees only the opener — 6 tokens:
`[CLS] (sB aY lB eY tB)`.

### `CLS` token
A single `CLS` token leads the sequence; its final hidden state is the pooled summary the
policy & value heads read. It also carries `TurnEmb[current_turn]`, so the model knows which
turn it is deciding → **how many guesses remain** (early = probe for info; near turn 6 = play
safe). `CLS` uses a dedicated `LetterEmb` entry and the `none` `FeedbackEmb` entry.

### No separators
Turn boundaries are implicit: every turn is exactly 5 cells, so `turn = cell_index // 5`,
`column = cell_index // 5`'s remainder. We make both explicit via `TurnEmb`/`ColEmb` rather
than inferring from a flat position, and we use **no `STEP`/`SEP` separator tokens** (the
earlier sketch's separators were redundant).

### Embedding tables (all tiny)
| Table | Entries | Params |
|---|---|---|
| `LetterEmb` | 26 letters + `CLS` + `PAD` ≈ 28 | ~7,168 |
| `FeedbackEmb` | `B`, `Y`, `G`, `none` (CLS/pad) = 4 | ~1,024 |
| `TurnEmb` | 6 (turns 1–6) | ~1,536 |
| `ColEmb` | 5 (columns 1–5) | ~1,280 |
| **Total** | | **~11K params (~0.17% of the model)** |

**Max sequence length = `CLS` + 5 turns × 5 cells = 26 tokens.** (The model plays turns 2–6,
so it sees at most 5 completed turns of context.)

## Reward function

The reward is the **only** thing that tells the model what "good play" means — it maximizes
whatever we give points for. Settled design (see `requirements.md` §3.10):

### Intuition
- **Win → big reward; lose → nothing.** Winning is worth *far* more than speed, so the model
  never gambles a winnable game just to finish a turn sooner.
- **Win fast → small extra bonus.** A gentle tiebreaker between two *safe* wins, nothing more.
- **"Getting warmer" hints during the game.** Small reward for guesses that narrow the set of
  still-possible answers — this gives useful signal *before* the first win (early on the model
  is near-random and almost never wins, so a win/lose-only signal barely teaches anything).
- **No punishments for "rule-breaking" guesses.** We do **not** penalize using a gray letter,
  moving a green, dropping a yellow, etc., and we do **not** trim the menu to consistent
  words. Those restrictions = **Hard Mode**, which *loses* the probe tactic and likely caps
  win rate below 100%.

#### Why Normal mode (full freedom) beats forced consistency — the probe example
Suppose you've narrowed it to 8 words, all ending `-OUND`, with 2 guesses left:
`BOUND FOUND HOUND MOUND POUND ROUND SOUND WOUND`. If forced to guess only *consistent*
words, you can test them only one at a time and **run out of turns**. The smart move is one
**probe** word cramming several of those first letters (`B,F,H,M,…`) to reveal the answer in a
single shot — but that probe isn't itself a candidate, so penalties / menu-trimming would
*forbid the move that wins.* This is why we keep the full 12,972-word action space. The
candidate-narrowing shaping rewards good probes automatically; a wasteful repeat narrows
nothing and earns nothing — so no explicit penalties are needed.

### Concrete spec
Let `k` = guesses used when solved (includes the fixed turn-1 opener; the model plays turns
2–6, so a model win has `k ∈ 2..6`).

- **Terminal:**
  - Win on guess `k`: `R_win = W_BASE + W_SPEED · (6 − k)` — defaults `W_BASE = 10`,
    `W_SPEED = 0.5` (slow win = 10, fast win ≈ 12). Tunable.
  - Loss: `R_loss = 0`.
  - Invariant: **win/loss gap (≈10) ≫ speed spread (≈2)** → winning dominates ~5:1.
- **Shaping** (per model step, turns 2–6) — the "warmer" hints:
  - `F = shaping_coef · (γ·Φ(s′) − Φ(s))`, with `Φ(s) = −log₂(|consistent answers|)`.
  - **Sizing — the key tunable.** Size it so a *full winning game's* warmer rewards sum to
    **≈ ⅓–½ of a win** (~3–5 with `W_BASE = 10`): meaningful enough to actually guide, but
    **never ≥ a win** (else the model may "narrow nicely but not close out"). Default
    `shaping_coef = 0.8` (≈ half a win). Raise it if early learning is sluggish; lower it if
    the model stalls just short of winning.
  - **Annealed to 0** over the back half of training, so the model finishes judged purely on
    winning — which makes erring generous *early* low-risk.
  - **Why moderate suffices (not huge):** early on wins are rare, so the warmer reward is the
    *only* signal that varies between games — it drives learning regardless of absolute size;
    and PPO normalizes advantages, so the *direction* ("narrowed more → better") matters more
    than raw magnitude.
  - **Potential-based** → provably cannot change the optimal policy for any coefficient
    (Ng et al. 1999). Credited only for the model's own guesses (the opener's narrowing is
    free context).
  - `|consistent answers|` is cheap to compute from a precomputed `feedback[guess, answer]`
    table.
- **Discount:** short episodes → `γ = 0.99` (or `1.0`).
- **Action space:** full 12,972 valid words every turn — **no masking, no consistency
  penalties** (Normal mode).
- **Exact repeats:** not masked — the reward already gives a repeat nothing (narrows nothing →
  zero shaping) and it wastes a turn, so the model learns to avoid it. Tracked via the
  `repeat_rate` diagnostic (see `logging.md`), not forbidden.
- **"Bad guess" behaviors → diagnostics, not penalties.** Repeats, reusing grays, ignoring
  greens, misplacing yellows, and inconsistent guesses are *logged* (per-turn `flags` +
  per-iteration rates), never penalized — the shaping reward already scores each guess by its
  informativeness, which is a better signal than any consistency rule.

Absolute scale isn't critical (PPO normalizes advantages); what matters is the **ordering**:
a **win (≈10)** is the top prize, a full game's **warmer rewards (≈3–5)** are a meaningful but
smaller chunk, and the **speed bonus (≈2)** is just a gentle tiebreaker between safe wins.

## Model initialization

We train from scratch, so "init" = setting the starting **random** weights. Goals: stable
signal flow through the stack, and a **broadly exploratory initial policy**. A fixed
**random seed** controls the draw (logged in `config.json`); init on CPU, then `.to("mps")`.

### Trunk — GPT-2 / nanoGPT scheme

| Component | Init |
|---|---|
| Linear weights (attn Q/K/V/out, FFN) | Normal, `mean=0, std=0.02` |
| Biases | `0` |
| Token / positional / slot embeddings | Normal, `std=0.02` |
| LayerNorm | weight `=1`, bias `=0` |
| **Residual-projection weights** (attn output proj + 2nd FFN linear) | scaled by `1/√(2·n_layers)` |

The residual scaling matters: every layer *adds* into the residual stream, so without
down-scaling the output projections the variance grows layer-by-layer. (nanoGPT pattern:
implemented via `model.apply(_init_weights)` plus a residual-scaling pass.)

### Heads — RL-specific

- **Policy head:** initialize with a **tiny gain (≈0.01)** → near-zero weights → the
  **initial policy is ≈ uniform** over all ~12,972 guesses (max entropy ≈ `log 12972 ≈ 9.5`
  nats). This makes the agent a broad explorer at the start of self-play instead of
  fixating early. Standard well-tuned-PPO practice.
- **Value head:** normal/small gain (≈1.0) → initial value estimates near a neutral
  baseline.

### Notes
- **Alternative:** orthogonal init (hidden gain `√2`, policy-output gain `0.01`, value-output
  gain `1.0`) is the other common PPO convention. Default here is normal (nanoGPT) trunk +
  small-gain policy head; orthogonal is an easy A/B if stability requires it.
- Init works **with** the entropy bonus + reward shaping + curriculum (see `requirements.md`
  §3.6): high starting entropy from init, kept from collapsing by the entropy bonus, with
  shaping/curriculum supplying early gradient — together this is what gets from-scratch
  exploration off the ground.

## M0 — Game scaffold & `play` (build scope & interfaces)

First milestone: a **correct, testable Wordle simulator + a watchable runner**. **No model,
no training.** numpy-only, plus `rich` for the UI. Everything later (model, PPO) plugs into
these interfaces. Run commands assume the working directory is `wm/` (its own venv:
`wm/.venv`).

### Modules
```
wm/wordle/  words.py        # load lists, word↔id maps, letter-id arrays
            feedback.py     # B/Y/G algorithm (single + batched), duplicate-correct
            env.py          # batched Wordle env (auto-plays opener on reset)
wm/agents/  base.py         # Agent protocol (the M1 model conforms to this)
            random_agent.py # baseline player
wm/         eval.py         # evaluate() → stats (quiet, batched; reused by play + training)
            play.py         # `play` runner: run N games, live showcase, summary report
            render.py       # board + report rendering (rich)
            config.py       # opener, paths, max_guesses, seed
wm/tests/   test_feedback.py  test_env.py
```

### Key interfaces
```python
# words.py
class WordList:                      # answers (2315), allowed (12972), id maps, letter-id arrays
def load_wordlist(data_dir=None) -> WordList            # defaults to wm/data

# feedback.py  (codes: 0=gray/B, 1=yellow/Y, 2=green/G)
def score(guess: str, secret: str) -> str               # "BYBYG" (human/tests)
def feedback_batch(guesses, secrets) -> np.ndarray      # [B,5] letter-ids → [B,5] codes

# env.py
@dataclass
class Obs: guesses; feedbacks; turn; done; won           # semantic board (not tokens)
class WordleEnv:
    def reset(self, secrets) -> Obs                      # plays opener; first action = turn 2
    def step(self, actions) -> tuple[Obs, reward, done, info]
    @property
    def all_done(self) -> bool

# agents/base.py
class Agent(Protocol):
    def act(self, obs: Obs) -> np.ndarray                # [B] guess word-ids

# eval.py
@dataclass
class EvalResult: win_rate; avg_guesses; guess_distribution; failed_words
def evaluate(agent, env, secrets=None) -> EvalResult
```
Single id space = the **12,972 allowed words** (action ids 0–12971). Secrets are sampled from
the 2,315 answers (their indices within `allowed`).

### The `play` runner
- CLI (from `wm/`): `python play.py --n 100 --agent random --seed 0 [--quiet] [--slow]`
- **Default 100 games** (configurable `--n`), random seeded sample of answers.
- **Live dashboard (rich):** current board with `🟩🟨⬛` tiles + running tally (games, win
  rate, avg), refreshing in place via `rich.Live`.
- **Final report (rich table):** win rate, guess-count distribution bars (1–6 + fail), avg
  guesses (wins only), failed-word list.
- Flags: `--quiet` = summary only; `--slow` = animate tile-by-tile.
- Built on the same env + agent; **`evaluate()` is the quiet batched stats engine** reused by
  training later (M2). `play` is the watchable face; `evaluate()` is the measurement engine.

### Definition of done
- `pytest` green from `wm/` — especially `test_feedback.py` duplicate cases (the key gate).
- `python play.py` runs the live showcase and prints the summary report; the random baseline
  shows a low (sanity-check) win rate.

### Build order within M0
1. `words.py` → 2. `feedback.py` **+ `test_feedback.py`** (gate) → 3. `env.py` **+
`test_env.py`** → 4. `agents/` (random) + `eval.py` → 5. `render.py` + `play.py` (rich UI).
Verify each before the next.
