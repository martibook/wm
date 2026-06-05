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
uv run play.py --n 100 --agent random --seed 0
```

| Flag | Meaning | Default |
|---|---|---|
| `--n` | number of games | `100` |
| `--agent` | which player (`random` for now) | `random` |
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
- [ ] push higher (more exploration) + the **full 12,972-vocab** generic game
- [ ] deferred: guess-behavior diagnostics + `games.jsonl` sampling

Run training: `uv run train.py --curriculum full` (resume: `--resume <ckpt>`).

> The free per-word head **memorized** (0% on the full set); making each word's score depend
> on its **letters** (letter-grounded head) is what unlocked generalization. Next: more
> exploration to push past ~78%, then the full-vocabulary game.
