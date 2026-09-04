"""Fit every evaluation weight at once against engine-scored positions.

    curl -s https://database.lichess.org/lichess_db_eval.jsonl.zst \
      | zstd -dc | head -200000 | uv run python training/tune_weights.py

Instead of A/B testing one constant at a time (40 games each, hours, and blind to
interactions), this extracts the RAW value of every term, then solves for the weights
that best predict a strong engine's score. One pass, all terms, minutes.
"""

import json
import sys
from pathlib import Path

import chess
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agent  # noqa: E402

CLAMP = 800.0        # cp beyond this is "winning"; the detail past it is noise for tuning


def features(board: chess.Board, colour: chess.Color) -> list[float]:
    """Each term's value with its weight divided out, so the fit can choose the weight."""
    raw_material = sum(
        v * (len(board.pieces(p, colour)) - len(board.pieces(p, not colour)))
        for p, v in agent.PIECE_VALUE.items()
    )
    place, _ = agent.placement_and_material(board, colour)
    return [
        float(raw_material),
        place / agent.PST_WEIGHT,
        float(agent.inactive_pieces(board, colour)),
        float(agent.fighting_for_central_squares(board, colour)),
        float(agent.support_central_structure(board, colour)
              - agent.support_central_structure(board, not colour)),
        float(agent.dim_knight(board, colour) / max(agent.DIM_RIM_PENALTY, 1)),
        float(agent.king_away_from_centre(board, colour)),
        float(agent.passed_pawns_score(board, colour)),
        float(agent.pawn_phalanx(board, colour) - agent.pawn_phalanx(board, not colour)),
    ]


NAMES = ["material", "PST", "inactive", "centre_fight", "support_centre",
         "dim_knight", "king_away", "passed_pawns", "pawn_shield"]


def main() -> None:
    rows, targets = [], []
    skipped = 0
    for line in sys.stdin:
        record = json.loads(line)
        board = chess.Board(record["fen"])
        if not board.is_valid():
            skipped += 1
            continue
        analysis = max(record["evals"], key=lambda e: int(e["depth"]))
        pv = analysis["pvs"][0]
        if "cp" not in pv:
            continue
        cp = pv["cp"] * (1 if board.turn == chess.WHITE else -1)
        rows.append(features(board, board.turn))
        targets.append(max(-CLAMP, min(CLAMP, float(cp))))

    x = np.array(rows, dtype=np.float64)
    y = np.array(targets, dtype=np.float64)
    print(f"{len(y):,} positions ({skipped:,} illegal skipped), {x.shape[1]} terms\n")

    weights, *_ = np.linalg.lstsq(x, y, rcond=None)
    predicted = x @ weights
    baseline = np.sqrt(np.mean(y**2))
    residual = np.sqrt(np.mean((y - predicted) ** 2))

    current = [agent.MATERIAL_WEIGHT, agent.PST_WEIGHT, -agent.INACTIVE_PENALTY, 1.0,
               agent.CENTRE_BONUS, -agent.DIM_RIM_PENALTY, 1.0, 1.0, agent.SHIELD_BONUS]
    print(f"  {'term':16}{'current':>10}{'fitted':>10}{'ratio':>9}")
    for name, now, fit in zip(NAMES, current, weights, strict=True):
        ratio = f"{fit / now:6.1f}x" if abs(now) > 1e-9 else "     -"
        print(f"  {name:16}{now:>10.2f}{fit:>10.2f}{ratio:>9}")
    print(f"\n  rms error: {baseline:.0f} cp with no model"
          f"  ->  {residual:.0f} cp with these weights")


if __name__ == "__main__":
    main()
