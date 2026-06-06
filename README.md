# wm — Wordle Model

Train a tiny **character-level transformer** to play Wordle via **reinforcement-learning
self-play**. Target: **100% win rate**. Runs entirely on a laptop (Apple Silicon / MPS).

Full design lives in `docs/`:
- `docs/requirements.md` — decisions log (what & why)
- `docs/design.md` — architecture, input format, reward, model init, **M0 scope**
- `docs/logging.md` — training-run logging schemas

## Setup

Managed with [uv](https://docs.astral.sh/uv/) (Python 3.12 + deps + commands). From the
`wm/` directory:

```bash
uv sync
```

This creates `.venv/` and installs everything pinned in `pyproject.toml` / `uv.lock`.
(Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`.)

## How to run

Run everything via `uv run` from the `wm/` directory (it keeps the env in sync first).

### Run the tests

```bash
uv run pytest
```

### Play — watch the game

Run `N` games and watch a live, colored showcase, ending with a summary report:

```bash
uv run play.py --n 100 --ckpt runs/<run>/checkpoints/latest.pt --seed 0
```

| Flag | Meaning | Default |
|---|---|---|
| `--n` | number of games | `100` |
| `--agent` | which player (`model` or `random`) | `model` |
| `--ckpt` | model checkpoint to load (for `--agent model`; untrained if omitted) | `None` |
| `--seed` | random seed (reproducible) | `0` |
| `--quiet` | summary only, no live showcase | off |
| `--slow` | animate tile-by-tile | off |

## Project status

**M0** — game scaffold + `play` runner, no ML — **done**:
- [x] word lists + duplicate-correct feedback **+ tests**
- [x] batched env + tests (`pytest` green — 14 tests)
- [x] random agent + `evaluate()`
- [x] `play` runner (rich live dashboard + report)

**M1** — char-transformer + PPO self-play — **works; tests green (28)**:
- [x] model (encoder + **letter-grounded head**), reward+shaping, rollout, PPO, curriculum, logging
- [x] learns on small sets (Stage A 50 words → 87%)
- [x] **generalizes to the full 2,315 answers** — letter-grounded head → **~78%** (free head: 0%)
- [x] **answer-pool game → ~98%** (2,315 answers, 2,315-word guess pool) after the exploration
      fix; run `20260605-105918`, peak at iter 2200.
- [~] **full 12,972-vocab generic game** (phase C) — *in progress*. Warm-started from the 98%
      checkpoint (`iter_2200.pt`) into a single full-vocab stage (`--answers 2315 --pool -1`);
      run `20260605-155725`. Greedy full-vocab eval climbed **56% → ~96%** within ~450 iters and
      still rising toward the restricted-pool ceiling.
- [ ] deferred: guess-behavior diagnostics + `games.jsonl` sampling

Run training:
- Full curriculum (A→B→C, cold): `uv run train.py --curriculum full`
- Full-vocab from a strong checkpoint (warm-start): `uv run train.py --resume <ckpt> --answers 2315 --pool -1 --iters <N>`
  (strip the checkpoint's `run_id` first so it writes a fresh run dir instead of clobbering the source run)

> The free per-word head **memorized** (0% on the full set); making each word's score depend
> on its **letters** (letter-grounded head) is what unlocked generalization. The exploration fix
> then took the answer-pool game to ~98%. The full 12,972-word vocab is the last gap: a model
> trained only on the restricted pool collapses to ~56% when allowed to guess any word, so
> phase C trains directly on the full action space — warm-starting from the 98% checkpoint
> recovers most of that performance fast.
