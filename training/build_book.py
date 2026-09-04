"""Build a Polyglot opening book from the Lichess Elite PGN archive.

    uv run python training/build_book.py --out weights/elite.bin --plies 16 --min-games 4

Provenance is the point: this book is generated from a named public dataset by a
script in the repo, so it is reproducible if anyone asks.
"""

import argparse
import collections
import io
import struct
import zipfile
from pathlib import Path

import chess
import chess.pgn
import chess.polyglot

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "Lichess_Games" / "lichess_elite_2025-11.zip"
PROMO = {chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}


def polyglot_move(board: chess.Board, move: chess.Move) -> int:
    """Polyglot encodes castling as king-takes-own-rook, not as the king's two-square hop."""
    to_square = move.to_square
    if board.is_castling(move):
        rank = 0 if board.turn == chess.WHITE else 7
        to_square = chess.square(7 if chess.square_file(move.to_square) == 6 else 0, rank)
    return (
        chess.square_file(to_square)
        | (chess.square_rank(to_square) << 3)
        | (chess.square_file(move.from_square) << 6)
        | (chess.square_rank(move.from_square) << 9)
        | (PROMO.get(move.promotion, 0) << 12)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "weights" / "elite.bin")
    parser.add_argument("--plies", type=int, default=16)
    parser.add_argument("--min-games", type=int, default=4)
    parser.add_argument("--max-games", type=int, default=200_000)
    arguments = parser.parse_args()

    counts: collections.Counter[tuple[int, int]] = collections.Counter()
    games = 0
    with zipfile.ZipFile(ARCHIVE) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".pgn"))
        print(f"reading {name}")
        with archive.open(name) as raw:
            stream = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
            while games < arguments.max_games:
                game = chess.pgn.read_game(stream)
                if game is None:
                    break
                games += 1
                board = game.board()
                for ply, move in enumerate(game.mainline_moves()):
                    if ply >= arguments.plies:
                        break
                    counts[(chess.polyglot.zobrist_hash(board), polyglot_move(board, move))] += 1
                    board.push(move)
                if games % 20_000 == 0:
                    print(f"  {games:,} games, {len(counts):,} entries")

    kept = {k: v for k, v in counts.items() if v >= arguments.min_games}
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    with arguments.out.open("wb") as handle:
        for (key, move), weight in sorted(kept.items()):
            handle.write(struct.pack(">QHHI", key, move, min(weight, 65535), 0))

    size = arguments.out.stat().st_size
    print(f"\n{games:,} games -> {len(counts):,} raw entries, {len(kept):,} kept "
          f"(seen >= {arguments.min_games} times)")
    print(f"wrote {arguments.out}  ({size:,} bytes)")


if __name__ == "__main__":
    main()
