"""Watch the trained model play Wordle in a real, VISIBLE browser (local board).

The model plays like a person: it types the fixed opener `salet`, reads the tile colors
off the rendered board, picks the next guess, and repeats — it never sees the answer.
The board (web/wordle.html) is fed the model's own word lists, so every guess is valid.

    uv run playwright install chromium      # one-time: fetch the browser binary
    uv run play_web.py --ckpt runs/<run>/checkpoints/optimal.pt
"""
from __future__ import annotations

import argparse

import numpy as np

from agents.model_agent import ModelAgent
from config import REPO_ROOT
from wordle.env import Obs
from wordle.words import load_wordlist, word_to_letter_ids

HTML = (REPO_ROOT / "web" / "wordle.html").as_uri()
OPENER = "salet"
STATE_CODE = {"absent": 0, "present": 1, "correct": 2}   # tile data-state -> feedback code


def blank_obs() -> Obs:
    return Obs(np.zeros((1, 6, 5), np.int8), np.zeros((1, 6, 5), np.int8),
               np.zeros(1, np.int64), np.zeros(1, bool), np.zeros(1, bool))


def play_game(page, agent, wl, slow) -> tuple[bool, int]:
    obs, word = blank_obs(), OPENER
    for turn in range(6):
        page.wait_for_timeout(slow)
        page.keyboard.type(word, delay=110)          # type it out, letter by letter
        page.keyboard.press("Enter")
        page.wait_for_function("(t) => window.WM.turn === t", arg=turn + 1)
        states = page.locator(f"#row-{turn} .tile").evaluate_all("els => els.map(e => e.dataset.state)")
        codes = np.array([STATE_CODE[s] for s in states], np.int8)   # read the colors off the screen
        obs.guesses[0, turn] = word_to_letter_ids(word)
        obs.feedbacks[0, turn] = codes
        obs.turn[0] = turn + 1
        if (codes == 2).all():
            return True, turn + 1
        word = wl.word_of(int(agent.act(obs)[0]))     # model picks the next guess
    return False, 6


def main() -> None:
    ap = argparse.ArgumentParser(prog="play_web", description="Watch the model play Wordle in a browser.")
    ap.add_argument("--ckpt", default="runs/20260605-155725/checkpoints/optimal.pt")
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None, help="answer index (omit = random each game)")
    ap.add_argument("--slow", type=int, default=700, help="ms pause between guesses")
    ap.add_argument("--headless", action="store_true", help="hide the browser (default: visible)")
    ap.add_argument("--record", default=None, help="dir to save a video of the session (.webm)")
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    wl = load_wordlist()
    agent = ModelAgent.from_checkpoint(args.ckpt, wl)
    inject = f"window.WM_ANSWERS={list(wl.answers)!r}; window.WM_ALLOWED={list(wl.allowed)!r};" \
             + ("" if args.seed is None else f" window.WM_SEED={args.seed};")
    viewport = {"width": 460, "height": 700}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        ctx = browser.new_context(viewport=viewport,
                                  **({"record_video_dir": args.record, "record_video_size": viewport}
                                     if args.record else {}))
        ctx.add_init_script(inject)
        page = ctx.new_page()
        wins = 0
        for g in range(args.games):
            page.goto(HTML)
            won, n = play_game(page, agent, wl, args.slow)
            wins += won
            print(f"game {g + 1}: {'WON in ' + str(n) if won else 'LOST'}  (answer {page.evaluate('window.WM.answer')})")
            page.wait_for_timeout(args.slow * 2)
        print(f"\n{wins}/{args.games} solved")
        ctx.close()      # finalizes the video file
        browser.close()


if __name__ == "__main__":
    main()
