"""Play the agent yourself. You get unlimited time; it gets the competition clock.

    uv run python test/play_human.py                 # you are White, standard start
    uv run python test/play_human.py --black         # you are Black
    uv run python test/play_human.py --fen "<FEN>"   # start from a given position

Enter moves as SAN (Nf3, exd5, O-O) or UCI (g1f3). Other commands:
    board   redraw       moves   list legal moves
    undo    take back your last move and the reply
    fen     print the current FEN
    quit
"""

import argparse
import sys
import time
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agent  # noqa: E402

BASE_MS = 120_000        # same as harness/rules.py
INCREMENT_MS = 500


def show(board: chess.Board, human: chess.Color, clock_ms: float) -> None:
    ranks = str(board if human == chess.WHITE else board.mirror()).split("\n")
    labels = range(8, 0, -1) if human == chess.WHITE else range(1, 9)
    print()
    for label, row in zip(labels, ranks, strict=True):
        print(f"   {label}  {row}")
    files = "a b c d e f g h" if human == chess.WHITE else "h g f e d c b a"
    print(f"      {files}")
    last = board.peek().uci() if board.move_stack else "-"
    print(f"\n   bot clock {clock_ms / 1000:6.1f}s    last move {last}")


def parse(board: chess.Board, text: str) -> chess.Move | None:
    for reader in (board.parse_san, board.parse_uci):
        try:
            return reader(text)
        except ValueError:
            continue
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--black", action="store_true", help="play as Black")
    parser.add_argument("--fen", default=None, help="starting position")
    arguments = parser.parse_args()

    board = chess.Board(arguments.fen) if arguments.fen else chess.Board()
    human = chess.BLACK if arguments.black else chess.WHITE
    clock = float(BASE_MS)

    print(__doc__.split("\n\n")[0])
    print(f"\nYou are {'Black' if human == chess.BLACK else 'White'}. "
          f"Bot has {BASE_MS / 1000:g}s + {INCREMENT_MS / 1000:g}s. You have as long as you like.")

    while True:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            show(board, human, clock)
            print(f"\n   GAME OVER: {outcome.termination.name.lower()}  ({outcome.result()})")
            return

        if board.turn == human:
            show(board, human, clock)
            text = input("\n   your move> ").strip()
            if text in ("quit", "q"):
                return
            if text == "board":
                continue
            if text == "fen":
                print(f"   {board.fen()}")
                continue
            if text == "moves":
                print("   " + ", ".join(sorted(board.san(m) for m in board.legal_moves)))
                continue
            if text == "undo":
                for _ in range(2):
                    if board.move_stack:
                        board.pop()
                continue
            move = parse(board, text)
            if move is None or move not in board.legal_moves:
                print(f"   '{text}' is not a legal move here. Type 'moves' to list them.")
                continue
            board.push(move)
        else:
            print("\n   bot thinking...", end="", flush=True)
            started = time.monotonic()
            uci = agent.get_move(board.fen(), int(clock))
            elapsed = (time.monotonic() - started) * 1000.0
            clock -= elapsed
            if clock < 0:
                show(board, human, clock)
                had = clock + elapsed
                print(f"\n   the bot flagged (used {elapsed:.0f}ms with {had:.0f}ms left)")
                return
            clock += INCREMENT_MS
            move = chess.Move.from_uci(uci)
            print(f" played {board.san(move)}  ({elapsed:.0f}ms)")
            board.push(move)


if __name__ == "__main__":
    main()
