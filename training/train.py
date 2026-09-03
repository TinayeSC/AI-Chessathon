"""Train a position-evaluation network on the Lichess engine-evaluation dump.

Usage:
    curl -s https://database.lichess.org/lichess_db_eval.jsonl.zst \
      | zstd -dc | head -200000 \
      | uv run python training/train.py

Reads JSON-lines on stdin, writes weights/eval_net.npz.
Nothing here ships. The agent loads the .npz with numpy and never imports torch.
"""

import json
import sys
from pathlib import Path

import chess
import numpy as np
import torch

# --- knobs ------------------------------------------------------------------
HIDDEN_1 = 256          # first hidden layer width. Bigger = smarter but slower at runtime.
HIDDEN_2 = 32
BATCH_SIZE = 1024       # positions per gradient step.
EPOCHS = 30             # full passes over the data.
LEARNING_RATE = 1e-3
VAL_FRACTION = 0.1      # held out to check we are learning, not memorising.
CP_SCALE = 400.0        # centipawns that map to ~0.76 after tanh. Standard-ish choice.
MAX_PIECES = 32         # most pieces on a board; the index array is padded to this.
PAD = 768               # sentinel index meaning "no piece"; dropped by expand().
OUT_PATH = Path(__file__).resolve().parent.parent / "weights" / "eval_net.npz"

# Order matters and must be identical in agent.py, or the net reads a scrambled board.
PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def encode_indices(board: chess.Board) -> np.ndarray:
    """Return the indices of the 768-vector that are 1, padded to 32 with PAD.

    The dense form is 12 piece-planes x 64 squares, one-hot:
      - planes 0-5  are MY pawn/knight/bishop/rook/queen/king
      - planes 6-11 are THEIR pawn/knight/bishop/rook/queen/king
      - when Black is to move the board is mirrored so "my back rank" is always rank 1

    One network then serves both colours, matching the sign convention negamax
    already uses (every score is from the mover's point of view).

    We store indices rather than the dense vector because only ~32 of the 768
    entries are ever set. Dense float32 costs 3072 bytes per position; 32 uint16
    indices cost 64. That is the difference between 1M and 10M positions in RAM.
    """
    out = np.full(MAX_PIECES, PAD, dtype=np.uint16)
    flip = board.turn == chess.BLACK
    for slot, (square, piece) in enumerate(board.piece_map().items()):
        # square_mirror flips rank 1 <-> rank 8, so Black sees the board as White does.
        sq = chess.square_mirror(square) if flip else square
        mine = piece.color == board.turn
        plane = PIECE_ORDER.index(piece.piece_type) + (0 if mine else 6)
        out[slot] = plane * 64 + sq
    return out


def expand(index_batch: np.ndarray) -> np.ndarray:
    """(B, 32) uint16 indices -> (B, 768) float32 one-hot.

    Scatter into a 769-wide buffer so the PAD index (768) has somewhere harmless
    to land, then drop that last column. Done per mini-batch, so only one batch
    is ever dense in memory.
    """
    rows = index_batch.shape[0]
    dense = np.zeros((rows, 769), dtype=np.float32)
    row_ids = np.repeat(np.arange(rows), index_batch.shape[1])
    dense[row_ids, index_batch.ravel()] = 1.0
    return dense[:, :768]


def load_positions() -> tuple[np.ndarray, np.ndarray]:
    """Read the eval dump from stdin and build the training arrays."""
    inputs: list[np.ndarray] = []   # one (32,) uint16 row per position
    targets: list[float] = []

    skipped = 0
    for line in sys.stdin:
        record = json.loads(line)
        board = chess.Board(record["fen"])          # 4-field FEN; python-chess fills the rest

        # The dump contains corrupt FENs that users submitted for analysis: boards
        # with 33+ pieces, and boards where the side NOT to move is in check (so a
        # king can be captured). is_valid() rejects both, and it is cheaper than
        # counting pieces because it works on bitboards.
        if not board.is_valid():
            skipped += 1
            continue

        # Each record holds several analyses; take the deepest one.
        analysis = max(record["evals"], key=lambda e: int(e["depth"]))
        principal = analysis["pvs"][0]              # pvs[0] is the engine's best line
        if "cp" not in principal:
            continue                                # mate scores live on a different scale; skip

        # The dump reports cp from WHITE's point of view. Our net speaks
        # side-to-move, so flip the sign when Black is on move.
        cp = principal["cp"] * (1 if board.turn == chess.WHITE else -1)

        inputs.append(encode_indices(board))
        # tanh squashes unbounded centipawns into (-1, 1). Without it a single
        # +9000 position would dominate the loss and the net would learn nothing else.
        targets.append(float(np.tanh(cp / CP_SCALE)))

    if not inputs:
        raise SystemExit("no positions read from stdin - is the pipe connected?")
    print(f"   skipped {skipped:,} illegal positions from the dump")
    return np.array(inputs, dtype=np.uint16), np.array(targets, dtype=np.float32)


def build_net() -> torch.nn.Sequential:
    """768 -> 256 -> 32 -> 1, with Tanh on the output to match the squashed target."""
    return torch.nn.Sequential(
        torch.nn.Linear(768, HIDDEN_1),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_1, HIDDEN_2),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_2, 1),
        torch.nn.Tanh(),
    )


def export(net: torch.nn.Sequential) -> None:
    """Save plain numpy arrays so the agent needs no torch at runtime.

    torch stores a Linear layer's weight as (out_features, in_features) because it
    computes x @ W.T. We want to write `x @ W` by hand in agent.py, so transpose here.
    Get this wrong and inference silently returns nonsense rather than erroring.
    """
    arrays: dict[str, np.ndarray] = {}
    for index, layer in enumerate([net[0], net[2], net[4]]):
        arrays[f"W{index}"] = layer.weight.detach().numpy().T.copy()
        arrays[f"b{index}"] = layer.bias.detach().numpy()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_PATH, **arrays)
    print(f"saved {OUT_PATH} ({OUT_PATH.stat().st_size:,} bytes)")
    for name, array in arrays.items():
        print(f"   {name}: {array.shape}")


def main() -> None:
    indices, labels = load_positions()
    print(f"{len(indices):,} positions loaded "
          f"({indices.nbytes / 1e6:.1f} MB as indices, "
          f"{len(indices) * 768 * 4 / 1e6:.0f} MB if it were dense)")

    # Shuffle once, then split. Without a held-out set you cannot tell learning
    # from memorising - training loss always goes down either way.
    rng = np.random.default_rng(0)
    order = rng.permutation(len(indices))
    indices, labels = indices[order], labels[order]
    split = int(len(indices) * (1 - VAL_FRACTION))
    train_idx, train_y = indices[:split], labels[:split]
    val_idx, val_y = indices[split:], labels[split:]
    print(f"   {len(train_idx):,} train / {len(val_idx):,} validation")

    torch.manual_seed(0)                      # reproducible runs
    torch.set_num_threads(1)
    net = build_net()
    optimiser = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    def batch_loss(idx_slice: np.ndarray, y_slice: np.ndarray) -> torch.Tensor:
        """Expand indices to dense for just this batch, then forward."""
        x = torch.from_numpy(expand(idx_slice))
        y = torch.from_numpy(y_slice).unsqueeze(1)   # (B,) -> (B,1) to match output
        return torch.nn.functional.mse_loss(net(x), y)

    for epoch in range(EPOCHS):
        net.train()
        # Reshuffle each epoch so mini-batches differ between passes.
        shuffle = rng.permutation(len(train_idx))
        running = 0.0
        for start in range(0, len(train_idx), BATCH_SIZE):
            picks = shuffle[start:start + BATCH_SIZE]
            # These four lines, in this order, are the whole of gradient descent.
            optimiser.zero_grad()                       # 1. clear previous gradients
            loss = batch_loss(train_idx[picks], train_y[picks])   # 2. forward + error
            loss.backward()                             # 3. which way should weights move
            optimiser.step()                            # 4. move them
            running += loss.item() * len(picks)

        net.eval()
        with torch.no_grad():                 # no gradients needed just to measure
            val_total = 0.0
            for start in range(0, len(val_idx), BATCH_SIZE):
                stop = start + BATCH_SIZE
                chunk = val_idx[start:stop]
                val_total += batch_loss(chunk, val_y[start:stop]).item() * len(chunk)
            val_loss = val_total / len(val_idx)
        # If train loss falls while validation rises, it is memorising - stop early.
        print(f"  epoch {epoch:3}  train {running / len(train_idx):.5f}  val {val_loss:.5f}")

    export(net)

    # Sanity check the sign convention. A side that is a queen up must score positive.
    net.eval()
    with torch.no_grad():
        start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        checks = [
            (start_fen, "startpos, expect ~0"),
            (start_fen.replace("rnbqkbnr", "rnb1kbnr"), "white up a queen, expect +"),
            (start_fen.replace("RNBQKBNR w", "RNB1KBNR b"), "black up a queen, expect +"),
        ]
        for fen, label in checks:
            row = encode_indices(chess.Board(fen))[None, :]
            value = net(torch.from_numpy(expand(row))).item()
            print(f"   {label:32} {value * CP_SCALE:+8.1f} cp")


if __name__ == "__main__":
    main()
