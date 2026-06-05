"""Pattern table + candidate counting correctness."""
import numpy as np

from wordle.feedback import feedback_batch, score
from wordle.feedback_table import encode_pattern, load_pattern_table
from wordle.words import load_wordlist

wl = load_wordlist()
PAT = load_pattern_table(wl)


def test_pat_matches_feedback_batch():
    rng = np.random.default_rng(1)
    gi = rng.integers(0, wl.n_allowed, 3000)
    ai = rng.integers(0, wl.n_answers, 3000)
    codes = feedback_batch(wl.allowed_ids[gi], wl.answers_ids[ai])
    assert (encode_pattern(codes) == PAT[gi, ai]).all()


def test_candidate_count_matches_bruteforce():
    g = wl.id_of("salet")
    a_idx = wl.answers.index("crane")
    obs = PAT[g, a_idx]
    via_table = int((PAT[g, :] == obs).sum())
    ref = score("salet", "crane")
    brute = sum(1 for a in wl.answers if score("salet", a) == ref)
    assert via_table == brute
