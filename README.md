# wm — Wordle Model

Train a tiny **character-level transformer** to play Wordle via **reinforcement-learning
self-play**. Target: **100% win rate**. Runs entirely on a laptop (Apple Silicon / MPS).

Full design lives in `docs/`:
- `docs/requirements.md` — decisions log (what & why)
- `docs/design.md` — architecture, input format, reward, model init, **M0 scope**
- `docs/logging.md` — training-run logging schemas

## Setup

Python 3.12. From the `wm/` directory:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

(Skip this if `.venv/` already exists.)

## How to run

All commands are run from the `wm/` directory. You can either prefix with `.venv/bin/python`
(shown below) or activate the venv first with `source .venv/bin/activate`.

### Run the tests

```bash
.venv/bin/python -m pytest
```

### Play — watch the game

Run `N` games and watch a live, colored showcase, ending with a summary report:

```bash
.venv/bin/python play.py --n 100 --agent random --seed 0
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

**M1+** — char-transformer, PPO self-play, training, eval. See `docs/`.

> The random agent wins ~0% (you must guess the exact word) — that's the expected
> baseline confirming the harness works. Real win rate comes once the model learns (M1+).
