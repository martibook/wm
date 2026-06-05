# Training-run logging / learning history

How we record the **learning history** during RL training. Goal: a **generic, structured,
human-readable, model-agnostic** record — useful for plotting learning curves, debugging
specific failures, and reproducing runs.

> **Status (M1).** Built: `config.json`, `metrics.jsonl`, `eval.jsonl`. **Planned / not yet
> built:** `games.jsonl` sampling and the guess-behavior **`diagnostics`** (and per-turn
> `flags`). Those sections below are the *intended* schema, not current output (the
> `games.jsonl` file is created but stays empty). The live `metrics.jsonl`/`config.json`
> schemas are shown with the actual fields below.

## Principles

- **Semantic, not tokenized.** Records use human-meaningful values (`secret: "crane"`,
  `feedback: "BBYBG"`, `reward: 1.6`) — **never** the integer token IDs we feed the model.
  Logs must be understandable without knowing the model's internals.
- **Generic format: JSONL.** One self-describing JSON object per line — language-agnostic,
  append-only, streamable, queryable with `jq` / pandas / DuckDB. CSV mirror optional for
  flat scalar metrics; Parquet is a scale option if game volume grows large.
- **Keyed by `iter`** (the PPO update number) so all streams join cleanly.
- **Don't log every game.** Self-play produces thousands of games per iteration and discards
  them. Log *all* iteration metrics and eval results (cheap scalars); log only a *sampled
  subset* of full game transcripts, plus the full failed-word list at eval.

## Feedback encoding

Feedback is recorded as a 5-char string over `{B, Y, G}` (gray / yellow / green), e.g.
`"BBYBG"`. Optionally a parallel `feedback_codes` array `[0,0,1,0,2]`. This is the *semantic*
game feedback, independent of the model's token vocabulary.

## Streams & schemas

One folder per run:

```
wm/runs/<run_id>/
  config.json     # run configuration (once)
  metrics.jsonl   # per PPO update
  eval.jsonl      # per full-sweep eval
  games.jsonl     # sampled semantic game transcripts
  checkpoints/    # model weights (not logs)
```

### 1. `config.json` — once per run (reproducibility)
The run config (a dump of the `RunConfig` dataclass + run metadata + the curriculum phases).
```json
{
  "seed": 0, "total_iters": 4800, "eval_every": 50, "ckpt_every": 100, "games_sample": 16,
  "model": {"d_model": 256, "n_layers": 4, "n_heads": 4, "d_ff": 1024, "dropout": 0.0},
  "ppo": {"batch_games": 2048, "epochs": 4, "minibatch": 4096, "lr": 0.0003, "clip": 0.2,
          "value_coef": 0.5, "entropy_coef": 0.01, "gamma": 0.99, "gae_lambda": 0.95,
          "kl_target": 0.02, "grad_clip": 0.5},
  "reward": {"win_base": 10.0, "win_speed": 0.5, "loss": 0.0,
             "shaping_coef": 0.8, "anneal_shaping": true},
  "curriculum": {"stage": "A", "n_answers": 200, "guess_pool_size": 800,
                 "opener": "salet", "max_guesses": 6},
  "run_id": "20260604-185351", "git_commit": "d90a93b", "device": "mps",
  "started": "2026-06-04T...Z",
  "phases": [{"name": "A", "answers": 200, "pool": 800, "iters": 300},
             {"name": "B", "answers": 2315, "pool": 2315, "iters": 1500},
             {"name": "C", "answers": 2315, "pool": -1, "iters": 3000}]
}
```

### 2. `metrics.jsonl` — one record per PPO update (the learning curve)
Actual schema:
```json
{"iter": 1234, "phase": "B", "wall_clock": "2026-06-04T15:30:00Z", "env_steps": 9876543,
 "games": 2048, "n_answers": 2315, "pool": 2315,
 "train": {"win_rate": 0.91, "avg_guesses": 4.3, "mean_return": 1.2, "mean_reward": 0.3},
 "ppo": {"policy_loss": -0.012, "value_loss": 0.08, "entropy": 1.9, "approx_kl": 0.015,
         "clip_frac": 0.12, "explained_variance": 0.6, "grad_norm": 0.4},
 "hparams": {"lr": 3e-4, "entropy_coef": 0.01, "shaping_coef": 0.8},
 "iter_seconds": 0.8}
```
Field groups: `phase`/`n_answers`/`pool` = current curriculum stage; `train` = task
performance on the rollout; `ppo` = optimizer/RL diagnostics (watch `approx_kl`, `entropy`
for collapse, `explained_variance` for value-head health); `hparams` = current (possibly
annealed) values. *(The `diagnostics` block below is **planned, not yet emitted**.)*

#### Guess-behavior diagnostics (`diagnostics`) — PLANNED, not built
Per-rollout rates over the **model's own guesses** (turns 2–6), to understand/debug what the
model is learning. **Descriptive, not penalties** — they do **not** affect reward (we decided
not to forbid these; see `requirements.md` §3.10).
- `repeat_rate` — fraction that exactly repeat an earlier guess in the same game. The one
  always-wasteful move; should fall toward ~0 as the model learns (we rely on the reward to
  teach this, not masking — so this metric is how we confirm it actually does).
- `inconsistent_rate` — fraction that could not be the secret given the clues so far. **High
  is NOT bad** — intentional probes are inconsistent; read it alongside win rate, not as an
  error count.
- `reused_gray_rate` / `ignored_green_rate` / `misplaced_yellow_rate` — fraction that reuse a
  known-absent letter / fail to place a known green / put a known yellow back in a known-wrong
  spot (or drop a known yellow). Watch wasteful patterns shrink while smart probes persist.

Per-game detail is available via `flags` on each turn in `games.jsonl`.

### 3. `eval.jsonl` — one record per full 2,315-answer sweep
```json
{"iter": 1200, "wall_clock": "2026-06-04T15:28:00Z", "n_answers": 2315,
 "win_rate": 0.978, "avg_guesses": 3.9,
 "guess_distribution": {"1": 0, "2": 51, "3": 900, "4": 1100, "5": 210, "6": 3, "fail": 51},
 "failed_words": ["mamma", "fluff", "jazzy"]}
```
`failed_words` is the full list (the valuable debugging signal — e.g. spot double-letter
blind spots). This is the progress-to-100% curve.

### 4. `games.jsonl` — sampled game transcripts (semantic) — PLANNED, not built
*(The file is currently created but stays empty; the schema below is the intended design.)*
Sample ~N games/iteration (e.g. 16). Each record is one game.
```json
{"iter": 1200, "phase": "train", "secret": "crane", "won": true, "num_guesses": 3,
 "return": 1.6,
 "turns": [
   {"t": 1, "guess": "salet", "feedback": "BBYBG", "reward": 0.1,
    "candidates_after": 42, "is_opener": true},
   {"t": 2, "guess": "blare", "feedback": "BBGYG", "reward": 0.2,
    "candidates_after": 3, "logprob": -3.1, "value": 1.0, "entropy": 1.5, "is_opener": false,
    "flags": ["inconsistent"]},
   {"t": 3, "guess": "crane", "feedback": "GGGGG", "reward": 1.6,
    "candidates_after": 1, "logprob": -0.8, "value": 1.5, "entropy": 0.4, "is_opener": false,
    "flags": []}]}
```
Per-turn fields are semantic gameplay + lightweight policy diagnostics (`logprob`, `value`,
`entropy`). The fixed opener (turn 1) is flagged `is_opener` and is not a learned action.
`candidates_after` = number of answers still consistent after this feedback (handy for
spotting good/bad information gain). `flags` = any guess-behavior tags that apply this turn
(`repeat`, `inconsistent`, `reused_gray`, `ignored_green`, `misplaced_yellow`) — descriptive,
not penalties; see the `diagnostics` section above.

## Consumption

- Plot learning curves: read `metrics.jsonl` / `eval.jsonl` with pandas; `iter` on x-axis.
- Debug failures: filter `eval.jsonl` `failed_words`, then inspect matching `games.jsonl`.
- Reproduce: `config.json` carries seed + git commit + all hyperparameters.
- Optional: mirror scalar metrics to TensorBoard/CSV for live viewing (does not replace the
  canonical JSONL).
