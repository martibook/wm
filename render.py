"""Rich rendering for the `play` runner: Wordle board, live dashboard, summary report."""
from __future__ import annotations

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Real Wordle tile colors.
TILE_STYLE = {
    2: "bold white on #6aaa64",   # green
    1: "bold white on #c9b458",   # yellow
    0: "bold white on #787c7e",   # gray
}


def board_text(guesses, feedbacks, n_turns: int) -> Text:
    """Render the played rows as colored letter tiles."""
    t = Text()
    for r in range(n_turns):
        for c in range(5):
            ch = chr(int(guesses[r, c]) + 97).upper()
            t.append(f" {ch} ", style=TILE_STYLE[int(feedbacks[r, c])])
            t.append(" ")
        if r == 0:
            t.append("  opener", style="dim")
        t.append("\n")
    return t


def _running(games) -> tuple[int, float, float]:
    wins = [g for g in games if g.won]
    wr = len(wins) / len(games) if games else 0.0
    avg = sum(g.num_guesses for g in wins) / len(wins) if wins else 0.0
    return len(wins), wr, avg


def dashboard_panel(board, game_idx, n_games, games_so_far, done, secret_word, won) -> Panel:
    """Current game board + running tally (over games completed so far)."""
    wins, wr, avg = _running(games_so_far)
    header = Text(f" game {game_idx}/{n_games}", style="bold")
    if done:
        mark = "[green]✓[/]" if won else "[red]✗[/]"
        header.append_text(Text.from_markup(f"    answer {secret_word.upper()} {mark}"))
    else:
        header.append("    answer ·····", style="dim")
    stats = Text.from_markup(
        f" wins [cyan]{wins}[/]/{len(games_so_far)}  ([cyan]{wr:.0%}[/])    avg guesses [cyan]{avg:.2f}[/]"
    )
    return Panel(Group(header, Text(""), board, stats), title="play", border_style="blue")


def summary_renderable(result, agent_name: str, seed: int) -> Panel:
    wins = result.n - len(result.failed_words)
    wr_style = "green" if result.win_rate >= 0.5 else "yellow" if result.win_rate >= 0.2 else "red"

    keys = [k for k in result.distribution if k != "fail"] + ["fail"]
    maxc = max(list(result.distribution.values()) + [1])
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("solved in", justify="right")
    table.add_column("count", justify="right")
    table.add_column("", justify="left")
    for k in keys:
        c = result.distribution[k]
        bar = "█" * round(24 * c / maxc) if maxc else ""
        label = "✗ fail" if k == "fail" else str(k)
        style = "red" if k == "fail" else "green"
        table.add_row(label, str(c), Text(bar, style=style))

    shown = ", ".join(result.failed_words[:15])
    more = f"  (+{len(result.failed_words) - 15} more)" if len(result.failed_words) > 15 else ""
    failed_line = Text.from_markup(
        f"[dim]Failed ({len(result.failed_words)}): {shown}{more}[/]"
        if result.failed_words else "[green]No failures \U0001f389[/]"
    )

    bar_w = 24
    filled = round(bar_w * result.win_rate)
    win_bar = Text()
    win_bar.append("█" * filled, style="green")          # wins
    win_bar.append("█" * (bar_w - filled), style="red")   # failures
    win_line = Text()
    win_line.append("Win rate    ")
    win_line.append(f"{result.win_rate:.1%}", style=wr_style)
    win_line.append("  ")
    win_line.append_text(win_bar)

    body = Group(
        win_line,
        Text(f"Avg guesses {result.avg_guesses:.2f}   (wins only)"),
        Text(""),
        table,
        Text(""),
        failed_line,
    )
    return Panel(body, title=f"play · {agent_name} · {result.n} games · seed {seed}",
                 border_style="blue")
