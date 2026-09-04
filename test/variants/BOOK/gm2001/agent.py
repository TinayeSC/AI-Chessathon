"""The submission entrypoint. The platform imports this file and calls get_move."""


import math
import time
from pathlib import Path

import chess
import chess.polyglot
import numpy as np

DEPTH = 2
BOOK_PATH = Path(__file__).resolve().parents[4] / "weights" / "gm2001.bin"
BOOK = chess.polyglot.open_reader(BOOK_PATH) if BOOK_PATH.exists() else None

### TRY WORKING ON DYNAMIC PIECE VALUE. PIECE VALUE STARTS AT STANDARD VALUES, BUT AS THE GAME
#PROGRESSES PIECES CAN GAIN OR LOSE VALUE

global castled 
#____________________________OPPENNING STRATEGY___________________________________________
PIECE_VALUE = {chess.PAWN: 100, 
               chess.KNIGHT: 310, 
               chess.BISHOP: 330, 
               chess.ROOK: 500, 
               chess.QUEEN: 900}
BISHOP_PAIR_BONUS = 2
CENTRE_BONUS = 5
PAWN_BONUS = 2 #Multiplier for having good central pawn structure
INACTIVE_PENALTY = 10
FIANCHETTO_BONUS = 3
DIM_RIM_PENALTY = 3
OPPONENT_KING_BONUS = 20
KING_ATTACK_BONUS = 0
# Import time runs once per game, inside a 60 second budget, before your clock starts.
# Load weights and build tables out here, not inside get_move.

FIANCHETTOS: dict[chess.Color, tuple[tuple[chess.Square, chess.Square], ...]] = {
      chess.WHITE: ((chess.B2, chess.B3), (chess.G2, chess.G3)),
      chess.BLACK: ((chess.B7, chess.B6), (chess.G7, chess.G6)),
  }



HAS_CASTLED: dict[chess.Color, tuple[tuple[chess.Square, chess.Square], ...]] = {
      chess.WHITE: ((chess.G1, chess.H1), (chess.C1, chess.D1)),
      chess.BLACK: ((chess.G8, chess.H8), (chess.C8, chess.D8)),
  }


CENTRAL_PAWNS_GUARDS: dict[chess.Color,tuple[tuple[chess.Square,chess.Square],...]] = {

     chess.WHITE: ((chess.E4, chess.F3), (chess.D4, chess.C3),
                  (chess.E4, chess.D3), (chess.D4, chess.E3)),
     chess.BLACK: ((chess.E5, chess.F6), (chess.D5, chess.C6),
                  (chess.D5, chess.E6), (chess.E5, chess.D6)),
}


KING_STARTING_SQUARE: dict[chess.Color,chess.Square]= {
    chess.WHITE : chess.E1,
    chess.BLACK: chess.E8
}

KING_CASTLING: dict[chess.Color, list[chess.Square]] = {
    chess.WHITE : [chess.G1,chess.C1],
    chess.BLACK: [chess.G8,chess.C8]
}


CENTRE_AND_KNIGHTS: dict[chess.Color,tuple[tuple[chess.Square,chess.Square],...]] = {
    chess.WHITE:((chess.E4,chess.C3),(chess.D4, chess.F3),(chess.D4, chess.E2),(chess.E4,chess.D2)),
    chess.BLACK: ((chess.E5,chess.C6), (chess.D5,chess.F6),(chess.D5,chess.E7),(chess.E5,chess.D7))

}

CENTRE_AND_BISHOPS: dict[chess.Color,tuple[tuple[chess.Square,chess.Square],...]] = {
    chess.WHITE:((chess.E4,chess.D3),(chess.E4, chess.G2),(chess.D4, chess.E3),(chess.D4,chess.B2)),
    chess.BLACK: ((chess.E5,chess.G7), (chess.D5,chess.B7),(chess.D5,chess.E6),(chess.E5,chess.D6))
}

CENTRE_AND_ROOKS: dict[chess.Color,tuple[tuple[chess.Square,chess.Square],...]] = {
    chess.WHITE:((chess.E4,chess.E1),(chess.D4, chess.D1)),
    chess.BLACK: ((chess.E5,chess.E8), (chess.D5,chess.D8))
}

KNIGHT_ON_THE_RIM_IS_DIM: dict[chess.Color, list[chess.Square]] = {
    chess.WHITE:[chess.A3,chess.H3],
    chess.BLACK: [chess.A6,chess.H6]
}



CENTRE_SQUARES: dict[chess.Color,list[chess.Square]] = {
     chess.WHITE:[chess.C5,chess.D5,chess.E5,chess.F5],
     chess.BLACK:[chess.C4,chess.D4,chess.E4,chess.F4],
}

#WILL DISINCENTIVISE MOVING BACK TO STARTING POSITIONS
PIECES_HAVING_MOTION: dict[chess.Color, tuple[tuple[chess.PieceType, chess.Square], ...]] = {
  chess.WHITE:(
       (chess.BISHOP,chess.C1),
       (chess.BISHOP, chess.F1),
       (chess.KNIGHT,chess.B1),
       (chess.KNIGHT,chess.G1),
       (chess.PAWN,chess.E2),
       (chess.PAWN,chess.D2)),
    #    (chess.QUEEN,chess.D1),
    #    (chess.ROOK,chess.A1),
    #    (chess.ROOK,chess.H1)),

  chess.BLACK:(
       (chess.BISHOP,chess.C8),
       (chess.BISHOP, chess.F8),
       (chess.KNIGHT,chess.B8),
       (chess.KNIGHT,chess.G8),
       (chess.PAWN,chess.E7),
        (chess.PAWN,chess.D7)),
    #    (chess.QUEEN,chess.D8),
    #    (chess.ROOK,chess.A8),
    #    (chess.ROOK,chess.H8)),
}

MATE = 10**6




CP_SCALE = 400.0
PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

_w = np.load(Path(__file__).resolve().parents[4] / "weights" / "eval_net.npz")
W0, b0, W1, b1, W2, b2 = _w["W0"], _w["b0"], _w["W1"], _w["b1"], _w["W2"], _w["b2"]
# Read the training scale from the file itself. Keeping it as a separate constant
# here means a retrain at a different scale silently mis-scales every evaluation.
if "cp_scale" in _w.files:
    CP_SCALE = float(_w["cp_scale"][0])


def encode(board: chess.Board) -> np.ndarray:
    x = np.zeros(768, dtype=np.float32)
    flip = board.turn == chess.BLACK
    for sq, piece in board.piece_map().items():
        s = chess.square_mirror(sq) if flip else sq
        plane = PIECE_ORDER.index(piece.piece_type) + (0 if piece.color == board.turn else 6)
        x[plane * 64 + s] = 1.0
    return x


def nnue_eval(board: chess.Board) -> float:
    h = np.maximum(encode(board) @ W0 + b0, 0)
    h = np.maximum(h @ W1 + b1, 0)
    return float(np.tanh(h @ W2 + b2)[0]) * CP_SCALE

NET_WEIGHT = 0.2      # 0.0 = hand only, 1.0 = net only
FAST_EVAL = False

def leaf_eval(board: chess.Board, mover: chess.Color) -> float:
    net = nnue_eval(board)                      # centipawns, stdev ~255
    if FAST_EVAL:
        return net
    hand = float(evaluate_board(board, mover))   # bring stdev ~2466 down to ~247
    return NET_WEIGHT * net + (1.0 - NET_WEIGHT) * hand


def material(board: chess.Board, side: chess.Color) -> float:
    return sum(
        value * (len(board.pieces(piece, side)) - len(board.pieces(piece, not side)))
        for piece, value in PIECE_VALUE.items()
    )*5.0


# THIS IS EXPENSIVE ITERATING OVER 8 PAWNS. WE ONLY REALLY CARE ABOUT THE TWO
# CENTRAL ONES, AND MAYBE B AND G
# def pawn_structure(board: chess.Board, color: chess.Color) -> int:
#     return sum(
#               board.piece_type_at(square1) == chess.PAWN and board.color_at(square1) == color
#               and board.piece_type_at(square2) == chess.PAWN and board.color_at(square2) == color
#               for square1, square2 in CENTRAL_PAWNS_GUARDS[color]
#           )

# def central_pawn_chain(board: chess.Board, color: chess.Color) -> int:
#     return PAWN_BONUS * (pawn_structure(board, color) - pawn_structure(board, not color))


def support_central_structure(board: chess.Board, color: chess.Color)-> int:
       """Count color's pieces supporting its centre squares"""
       pawns = board.pieces(chess.PAWN,color)
       knights =  board.pieces(chess.KNIGHT,color)
       bishops = board.pieces(chess.BISHOP,color)
    #    queen = board.pieces(chess.QUEEN,color)
       rooks = board.pieces(chess.ROOK,color)

       support = 0 
       support += sum(pawn in pawns and knight in knights
                      for pawn, knight in CENTRE_AND_KNIGHTS[color])
       support += sum(pawn in pawns and bishop in bishops
                      for pawn, bishop in CENTRE_AND_BISHOPS[color])
       support += sum(pawn in pawns and rook in rooks for pawn,rook in CENTRE_AND_ROOKS[color] )
       return support 

def support_centre(board: chess.Board, color: chess.Color)-> int:
     return CENTRE_BONUS * (
         support_central_structure(board, color)
         - support_central_structure(board, not color)
     )


def real_defenders(board: chess.Board, color: chess.Color, square: chess.Square) -> int:
    return sum(1 for sq in board.attackers(color, square)
               if not board.is_pinned(color, sq))

     

def inactive_pieces(board: chess.Board, color: chess.Color) -> int:
    """Counts a colors inactive pieces: checks how many pieces are on their starting square"""
    return sum(
          board.piece_type_at(square) == piece_type and board.color_at(square) == color
          for piece_type, square in PIECES_HAVING_MOTION[color]
      )


def dim_knight(board:chess.Board,color:chess.Color)-> int:
    return -sum(
        board.piece_type_at(square) == chess.KNIGHT
        for square in KNIGHT_ON_THE_RIM_IS_DIM[color]
    )* DIM_RIM_PENALTY

def underdevloped(board: chess.Board, color: chess.Color) -> int:
    return - ( INACTIVE_PENALTY * inactive_pieces (board, color))


def hanging_pieces(board: chess.Board, color: chess.Color) -> None:
    pass

def fighting_for_central_squares(board:chess.Board,color:chess.Color) -> int:
    return sum(real_defenders(board,color,sq)
               for sq in CENTRE_SQUARES[color]) - sum(real_defenders(board,not color,sq)
                                                      for sq in CENTRE_SQUARES[not color])

def bishop_pair(board: chess.Board, color: chess.Color) -> int:
    mine = len(board.pieces(chess.BISHOP, color)) >= 2
    theirs = len(board.pieces(chess.BISHOP, not color)) >= 2
    return BISHOP_PAIR_BONUS * (mine - theirs)


  
def king_away_from_centre(board: chess.Board, color: chess.Color)-> int:
    return sum(
        board.piece_type_at(square) == chess.KING
        for square in KING_CASTLING[color] 
    )

def score_legal_captures(board: chess.Board) -> float:
   
   
    valuable_captures = 0.0
    for m in board.generate_legal_captures():

        attacker = board.piece_type_at(m.from_square) or chess.PAWN
        victim = chess.PAWN if board.is_en_passant(m) else (
            board.piece_type_at(m.to_square) or chess.PAWN
        )
        attacker_value = PIECE_VALUE[attacker]
        victim_value = PIECE_VALUE[victim]

        opponent_defenders =  real_defenders(board,not board.turn,m.to_square)

        if attacker_value > victim_value and opponent_defenders > 0:
            valuable_captures -=  0.1*(attacker_value - victim_value)

        elif attacker_value <victim_value:
                    valuable_captures +=  0.1*(victim_value-attacker_value)

        elif opponent_defenders < real_defenders(board,board.turn,m.to_square)-1:
            valuable_captures += 2

        else:
            valuable_captures += 1

    return valuable_captures


def king_zone_pressure(board: chess.Board, color: chess.Color) -> int:
    """Count how many times color attacks the squares around the enemy king."""
    enemy_king = board.king(not color)
    if enemy_king is None:
        return 0
    return sum(
        chess.popcount(board.attackers_mask(color, square))
        for square in board.attacks(enemy_king)
    )


def king_attack(board: chess.Board, color: chess.Color) -> int:
    return KING_ATTACK_BONUS * (
        king_zone_pressure(board, color) - king_zone_pressure(board, not color)
    )  

SHIELD_MASK: dict[chess.Color, list[int]] = {chess.WHITE: [], chess.BLACK: []}
for side in (chess.WHITE, chess.BLACK):
    for king_square in chess.SQUARES:
        kf, kr = chess.square_file(king_square), chess.square_rank(king_square)
        mask = 0
        for f in (kf - 1, kf, kf + 1):
            if not 0 <= f <= 7:
                continue
            for step in (1, 2):
                r = kr + step if side == chess.WHITE else kr - step
                if 0 <= r <= 7:
                    mask |= chess.BB_SQUARES[chess.square(f, r)]
        SHIELD_MASK[side].append(mask)


SHIELD_BONUS = 12
def pawn_phalanx(board: chess.Board, color: chess.Color) -> int:
    king_square = board.king(color)
    if king_square is None:
        return 0
    return chess.popcount(SHIELD_MASK[color][king_square] & board.pieces_mask(chess.PAWN, color))


def king_shelter(board: chess.Board, color: chess.Color) -> int:
    return SHIELD_BONUS * (pawn_phalanx(board, color) - pawn_phalanx(board, not color))


PASSED_MASK: dict[chess.Color, list[int]] = {chess.WHITE: [], chess.BLACK: []}
for color in (chess.WHITE, chess.BLACK):
    for square in chess.SQUARES:
        file_index, rank_index = chess.square_file(square), chess.square_rank(square)
        mask = 0
        for f in (file_index - 1, file_index, file_index + 1):
            if not 0 <= f <= 7:
                continue                                   # off the edge of the board
            ranks = range(rank_index + 1, 8) if color == chess.WHITE else range(0, rank_index)
            for r in ranks:
                mask |= chess.BB_SQUARES[chess.square(f, r)]
        PASSED_MASK[color].append(mask)

PASSER_BONUS_BY_RANK = (0, 5, 10, 20, 40, 70, 120, 0)

def passed_pawns(board: chess.Board, color: chess.Color)->  int:
    enemy_pawns = board.pieces_mask(chess.PAWN, not color)
    return  sum(
        PASSER_BONUS_BY_RANK[
            chess.square_rank(sq) if color == chess.WHITE else 7 - chess.square_rank(sq)
        ]
        for sq in board.pieces(chess.PAWN, color)
        if not PASSED_MASK[color][sq] & enemy_pawns
    )


def passed_pawns_score(board:chess.Board, color: chess.Color) -> int:
        return ((passed_pawns(board,color)) - (passed_pawns(board, not color)))


def evaluate_board(board: chess.Board, mover: chess.Color) -> float:
    # with open("Log.txt","a") as f:
    #     if mover:
    #         f.writelines(f"""
    #     UNDERDEVELOPED: {underdevloped(board,mover)}
    #     CENTRE_FIGHT: {fighting_for_central_squares(board,mover)}
    #     FIANCHETTO: {fianchetto(board,mover)}
    #     MATERIAL: {material(board, mover)}
    #     CENTRE_PAWN: {central_pawn_chain(board,mover)}
    #     CENTRE_SUPPORTS: {support_centre(board,mover)}
    #     KING_SAFETY: {king_safety(board,mover)}
    #     DIM_KNIGHT: {dim_knight(board,mover)}
    #     KING_AWAY_CENTRE: {king_away_from_centre(board,mover)}
    #     LEGAL_CAPTURES: {score_legal_captures(board)}
    #     # ACTIVITY: {increases_activity(before,board)}
    #     MOBILE:  {(0.25 * mobility)}\n"""
    # )

    #     f.close()

   
    
    check_fee = 0.0
    if board.is_check():
        check_fee+=25.0

   

    return (
    underdevloped(board,mover)+
    fighting_for_central_squares(board,mover)+
    # fianchetto(board,mover)+
    material(board, mover)+
    # central_pawn_chain(board,mover)+
    support_centre(board,mover)+
    # king_safety(board,mover)+
    dim_knight(board,mover)+
    king_away_from_centre(board,mover)+
    king_attack(board,mover)+
    check_fee+
    passed_pawns_score(board,mover)+
    king_shelter(board,mover)
    # score_legal_captures(board)+
    # increases_activity(before,board)+
   
   )

   
MAX_PLY = 24            # absolute ceiling across negamax AND quiesce combined
MAX_QUIESCE = 4         # extra plies quiesce may add once negamax stops
MAX_DEPTH = 8           # iterative deepening never asks for more than this


class TimeUp(Exception):
    """Raised deep in the search when the move budget is spent."""


NODES = 0
DEADLINE = 0.0


def check_clock() -> None:
    """Cheap clock check. time.monotonic() every node would cost more than it saves."""
    global NODES
    NODES += 1
    if NODES % 1024 == 0 and time.monotonic() > DEADLINE:
        raise TimeUp


def negamax(board: chess.Board, depth: int, alpha: float, beta:float, ply: int=0) -> float:
    check_clock()
    moves = list(board.legal_moves)
    if not moves:
        return -(MATE - ply) if board.is_check() else 0.0
    if depth <= 0 or ply >= MAX_PLY:
        return quiesce(board, alpha, beta, ply)

    best = float(-MATE)
    moves.sort(key=lambda m: not board.is_capture(m))
    for move in moves:
        board.push(move)
        extra = 1 if board.is_check() else 0
        score = -negamax(board, depth - 1 + extra, -beta, -alpha, ply + 1)
        board.pop()
        if score > best:
            best = score 
        if best > alpha:
             alpha = best
        if alpha >= beta:
             break
    return best

def quiesce(b: chess.Board, alpha: float, beta: float, ply: int = 0, qd: int = 0) -> float:
    check_clock()
    stand_pat = leaf_eval(b, b.turn)
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat
    if qd >= MAX_QUIESCE or ply >= MAX_PLY:
        return stand_pat
    
    for m in sorted(b.generate_legal_captures(),key=lambda m: -PIECE_VALUE[
        chess.PAWN if b.is_en_passant(m) else (b.piece_type_at(m.to_square) or chess.PAWN)
    ],
        ):
        # if real_defenders(b,not b.turn,m.to_square) > real_defenders(b,b.turn,m.to_square):
        # # if len(b.attackers(not b.turn, m.to_square)) > len(b.attackers(b.turn, m.to_square)):
        #     continue 
        
        b.push(m)
        score = -quiesce(b, -beta, -alpha, ply + 1, qd + 1)
        b.pop()
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha


SEEN: dict[int,int] = {}
LAST_MOVED:dict[tuple[chess.Square,chess.PieceType],int] = {}


def search_root(board: chess.Board, depth: int) -> tuple[chess.Move | None, float]:
    """One full-width pass at `depth`. Raises TimeUp if the budget runs out mid-pass."""
    best_move: chess.Move | None = None
    best_score = -math.inf
    for move in board.legal_moves:
        board.push(move)
        if board.is_checkmate():
            board.pop()
            return move, float(MATE)
        extra = 1 if board.is_check() else 0
        after = chess.polyglot.zobrist_hash(board)
        try:
            score = -negamax(board, depth - 1 + extra, -math.inf, -best_score, 1)
        finally:
            # pop before TimeUp unwinds, or the board is left corrupted for the
            # next iteration and every later move is generated from a wrong position.
            board.pop()

        seen = SEEN.get(after, 0)
        if seen >= 1:
            score = 50.0 - (25 * seen)

        lookup_key = (move.from_square, board.piece_type_at(move.from_square) or chess.PAWN)
        if lookup_key in LAST_MOVED:
            score -= 25.0 * LAST_MOVED[lookup_key]

        if score > best_score:
            best_move = move
            best_score = score
    return best_move, best_score


def get_move(fen: str, time_left_ms: int) -> str:
    global DEADLINE, NODES
    board = chess.Board(fen)

    if BOOK is not None:
        try:
            print(f"Book Move: {BOOK.weighted_choice(board).move}\n")
            return BOOK.weighted_choice(board).move.uci()
        except IndexError:
            pass

    # A pure fraction of what is left, never a fixed additive constant. Adding a
    # constant larger than the increment loses ground every move and eventually
    # flags - which is exactly what a fixed +300ms did at make gate's 5s + 0.1s.
    # A fraction converges: the clock settles where budget == increment.
    DEADLINE = time.monotonic() + max(time_left_ms / 25.0, 15.0) / 1000.0
    NODES = 0

    # Always hold a legal move, so a timeout at depth 1 still returns something.
    chosen_move = next(iter(board.legal_moves))
    best_score = -math.inf
    reached = 0

    for depth in range(1, MAX_DEPTH + 1):
        try:
            move, score = search_root(board, depth)
        except TimeUp:
            # negamax and quiesce push without try/finally (too costly in the hot
            # loop), so an unwind leaves their pushes on the board. Rebuild rather
            # than trying to unwind them: pushing onto a corrupted board is a crash.
            board = chess.Board(fen)
            break
        if move is None:
            break
        chosen_move, best_score, reached = move, score, depth
        if abs(score) >= MATE - MAX_PLY:
            break                                   # forced mate found, no point deeper

    board.push(chosen_move)
    key = (chosen_move.to_square, board.piece_type_at(chosen_move.to_square) or chess.PAWN)
    LAST_MOVED[key] = LAST_MOVED.get(key, 0) + 1
    digest = chess.polyglot.zobrist_hash(board)
    SEEN[digest] = SEEN.get(digest, 0) + 1
    board.pop()

    print(f"Played {chosen_move} depth {reached} score {best_score:.1f} nodes {NODES}", flush=True)
    return chosen_move.uci()



# def frankfurt_airport():
#      pass

# def fork():
#      pass

# def rook_open_file():
#      pass

# def strong_pawn_break():
#      pass


# def opposite_side_attack(board: chess.Board, color: chess.Color):
#     pass

# def threatened_pieces(board: chess.Board, color: chess.Color):
#     return board.is_attacked_by(color, chess.E5)

# def creating_threats(board: chess.Board, color: chess.Color):
#     pass

# def capitalise_on_advantage(board: chess.Board, color: chess.Color):
#     pass


# def king_safety(board: chess.Board, color: chess.Color)-> int:
#   k = board.king(color)  
#   if k is KING_STARTING_SQUARE[color]: 
#        return 0                              
#   else:
#        ring = board.attacks(k)                              
#        shield = ring & board.pieces(chess.PAWN, color)
#        return len(shield)

# def fianchettoed(board: chess.Board, color: chess.Color) -> int:
#     """Count color's fianchettoed bishops: on b2/g2 (b7/g7) behind its own knight pawn."""
#     bishops = board.pieces(chess.BISHOP, color)
#     pawns = board.pieces(chess.PAWN, color)
#     return sum(bishop in bishops and pawn in pawns for bishop, pawn in FIANCHETTOS[color])


# def fianchetto(board: chess.Board, color: chess.Color) -> int:
#     return FIANCHETTO_BONUS * (fianchettoed(board, color) - fianchettoed(board, not color))