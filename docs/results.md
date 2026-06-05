# Results / experiments log

Each entry records: **how the model was trained**, its **win rate over the answer vocabulary**
(guessing among the 2,315 answers), and its **win rate over the full vocabulary** (guessing
among all 12,972 words — the real generic game). Win rate = *greedy* eval over all 2,315
answers as secrets, opener fixed `salet`. Runs live in `runs/`.

> **The two win-rate columns are the two difficulty levels.** "Answer pool" = the model only
> has to choose among the 2,315 possible answers (easier, what we mostly train on). "Full
> vocab" = it must choose among all 12,972 words (the real game; a much bigger menu). The
> full-vocab number is the one that matters for the 100% target.

## Runs

| # | How trained | Win% — **answer pool** (2,315) | Win% — **full vocab** (12,972) | Run / notes |
|---|---|---|---|---|
| 1 | free per-word head · 50 secrets · pool 200 · ent 0.01 | ~56% (on the 50) | ~3% | memorizes; doesn't transfer. `runs/…181003` (old arch) |
| 2 | free head · curriculum A 200 → B 2,315 · ent 0.01 | A: ~87% → **B: ~7% (flat)** | — | A memorized → B crashed. deleted |
| 3 | free head · 2,315 · pool 2,315 · from scratch | **~0% (flat)** | — | free head can't learn the full set. deleted |
| 4 | letter-grounded · 100 secrets · pool 500 · ent 0.01 | 76% seen / **2% unseen** | ~low | small sets memorizable even with letter head. `runs/…224758` |
| 5 | **letter-grounded · 2,315 · pool 2,315 · ent 0.01, shaping 0.8→0, 2000 it** | **78.4%** | **42.3%** (avg 4.36) | first real generalization; entropy collapsed to ~0.3 → plateaued. `runs/20260604-232244` (iter_2000) |
| 6 | letter-grounded · 2,315 · pool 2,315 · **ent 0.03**, no-anneal, 3000 it | **~1.8% (abandoned @ it 490)** | — | **entropy too high** — stuck near max (7.16/7.75), policy stayed ~random, no learning. Run deleted. |
| 7 | letter-grounded · 2,315 · pool 2,315 · **ent 0.02**, no-anneal, 3000 it | **98.3%** (best @2200; final 97.8%) | **61.3%** | **beat #5 by +20pts (answers) / +19pts (full vocab)** — sustained exploration was the fix (0.01 collapsed; 0.03 too high). `runs/20260605-105918` |

(— = no loadable checkpoint to measure: deleted run, or incompatible old architecture.)

## Trajectories (greedy-eval win% over iters, answer pool)

The two letter-grounded full-2,315 runs, by entropy coefficient. **`0.01`** (run 232244,
finished) climbs faster early but its **entropy collapses (→0.3)** so it stalls ~48–52% for
iters 800–1200 before grinding to a 78% ceiling. **`0.02`** (run 105918, in progress)
**sustains exploration**, starts slower, then overtakes — at iter 1,000 it's **68.6% vs
48.7%**. (`0.03` is omitted: entropy stuck near max, never learned, ~1.8% — see row 6.)

| iter | `0.01` (ent → collapse) | `0.02` (ent sustained) |
|---|---|---|
| 100 | 0.4% | 0.4% |
| 200 | 1.0% | 1.0% |
| 300 | 2.1% | 1.9% |
| 400 | 4.0% | 2.5% |
| 500 | 11.4% | 7.0% |
| 600 | 28.6% | 11.5% |
| 700 | 39.4% | 20.6% |
| 800 | 42.0% | 37.1% |
| 900 | 52.4% | 46.7% |
| 1000 | 48.7% | **68.6%** |
| 1100 | 48.4% | 79.0% |
| 1200 | 57.0% | 86.4% |
| 1300 | 62.6% | 91.0% |
| 1400 | 69.5% | **93.5%** |
| 1500 | 69.0% | 95.2% |
| 1600 | 66.0% | 95.2% |
| 1700 | 66.8% | **96.6%** |
| 1800 | 73.9% | 96.3% |
| 1900 | 76.6% | 97.5% |
| 2000 | **78.0%** (best **78.4%** @1950) | **98.0%** |
| 2100 | — | 97.8% |
| 2200 | — | **98.3%** |
| 2300 | — | 98.1% |
| 2400 | — | 97.5% |
| 2500 | — | 97.1% |
| 2600 | — | 98.0% |
| 2700 | — | 98.0% |
| 2800 | — | 98.1% |
| 2900 | — | 97.4% |
| 3000 | — | 97.8% |

> **`0.01` ran 2,000 iters** (`—` above 2,000); **`0.02` ran 3,000** and plateaued ~98% after
> iter 2,200 (best **98.3%** @2200, final 97.8%). Full-vocab win rate of the `0.02` best
> checkpoint: **61.3%** (vs `0.01`'s 42.3%).

## Standing constraints / caveats
- **Answer-pool vs full-vocab** are the two columns above. Models are *trained* on the answer
  pool (2,315); the **full-vocab game is the untrained, harder target** — but note row 5
  already reaches **42%** full-vocab without ever training on it.
- **Hardware:** MacBook Pro M4 Max / 36 GB, PyTorch MPS. ~6–8 s/iter at batch 2,048.
- **Eval:** greedy (argmax) over all 2,315 answers as secrets; opener fixed `salet`.
- **Target:** 100% aspirational (measured on **full vocab**).
- **Entropy (exploration) sweet spot:** `0.01` collapses too fast → plateau (row 5); `0.03`
  is too high → policy never concentrates, no learning (row 6); `0.02` is the current middle
  ground under test (row 7). Diagnostic: watch whether entropy *drops steadily* (good) or
  *stays near max ~7.75* (too high).

## Key finding
The **letter-grounded head** enables generalization. The **free per-word head memorizes**
(~56% on a tiny set, but **~0–7% on the full answer set**, ~3% full-vocab). The
**letter-grounded head** reaches **78% answer-pool / 42% full-vocab** at default exploration
(row 5). **Sustained exploration** (entropy 0.01→0.02) then lifts it to **98.3% answer-pool /
61.3% full-vocab** (row 7) — the entropy collapse, not the architecture, was capping row 5.
Caveat (row 4): a *small* training set is memorizable regardless of head, so generalization
must be forced on a large set. See `docs/design.md` → Policy head.
