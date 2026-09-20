"""Verify engine.gpuchess against python-chess: perft counts + random-game differential test."""
import os, sys, random, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chess, torch
from engine import gpuchess as G
from engine.encoding import board_to_planes, move_to_index

dev = os.environ.get("CHESSAI_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

PERFT = [
    ("startpos", chess.STARTING_FEN, [20, 400, 8902, 197281]),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", [48, 2039, 97862]),
    ("pos3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", [14, 191, 2812, 43238]),
    ("pos4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", [6, 264, 9467]),
    ("pos5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", [44, 1486, 62379]),
    ("pos6", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", [46, 2079, 89890]),
]


def perft(st, depth, chunk=1024):
    total = 0
    for i in range(0, len(st), chunk):
        sub = st.index(torch.arange(i, min(i + chunk, len(st)), device=dev))
        mv = G.legal_moves(sub)
        if depth == 1:
            total += int(mv.n.sum())
            continue
        rows, cols = mv.valid.nonzero(as_tuple=True)
        child = G.make_moves(sub.index(rows), mv.code[rows, cols])
        total += perft(child, depth - 1, chunk)
    return total


def run_perft():
    ok = True
    for name, fen, counts in PERFT:
        st = G.state_from_boards([chess.Board(fen)], dev)
        for d, want in enumerate(counts, 1):
            t = time.time()
            got = perft(st, d)
            flag = "ok" if got == want else "MISMATCH"
            ok &= got == want
            print(f"perft {name} d{d}: {got} (want {want}) {flag} {time.time()-t:.1f}s", flush=True)
    return ok


def run_diff(n_games=60, max_plies=160, seed=1):
    rng = random.Random(seed)
    boards = [chess.Board() for _ in range(n_games)]
    ok = True
    checked = 0
    for ply in range(max_plies):
        live = [b for b in boards if not b.is_game_over(claim_draw=False)]
        if not live:
            break
        st = G.state_from_boards(live, dev)
        mv = G.legal_moves(st)
        term, val = G.terminal_info(st, mv)
        planes = G.to_planes(st).cpu().numpy()
        code = mv.code.cpu().numpy(); valid = mv.valid.cpu().numpy(); pol = mv.pol.cpu().numpy()
        for i, b in enumerate(live):
            ours = {int(c) for c, v in zip(code[i], valid[i]) if v}
            theirs = {G.move_to_code(m, b) for m in b.legal_moves}
            pols = {int(p) for p, v in zip(pol[i], valid[i]) if v}
            want_pols = {move_to_index(m) for m in b.legal_moves}
            if ours != theirs or pols != want_pols or int(mv.n[i]) != b.legal_moves.count():
                print("MOVE MISMATCH", b.fen(), sorted(ours ^ theirs)); ok = False
            if (planes[i] != board_to_planes(b)).any():
                print("PLANES MISMATCH", b.fen()); ok = False
            want_term = b.is_checkmate() or b.is_stalemate() or b.is_insufficient_material() or b.halfmove_clock >= 100
            if bool(term[i]) != want_term:
                print("TERMINAL MISMATCH", b.fen(), bool(term[i]), want_term); ok = False
            checked += 1
        # advance: play one random move on both sides, and check make_moves state equality
        choice = []
        for i, b in enumerate(live):
            m = rng.choice(list(b.legal_moves)) if b.legal_moves.count() else None
            choice.append(m)
        idx = [i for i, m in enumerate(choice) if m is not None]
        if idx:
            sub = st.index(torch.tensor(idx, device=dev))
            codes = torch.tensor([G.move_to_code(choice[i], live[i]) for i in idx], dtype=torch.int32, device=dev)
            new = G.make_moves(sub, codes)
            for j, i in enumerate(idx):
                live[i].push(choice[i])
                ref = G.state_from_boards([live[i]], dev)
                for f in ("sq", "stm", "castle", "ep", "half"):
                    if not torch.equal(getattr(new, f)[j], getattr(ref, f)[0]):
                        print("MAKE MISMATCH", f, live[i].fen()); ok = False
    print(f"differential: {checked} positions checked, {'ok' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    a = run_diff()
    b = run_perft() if "--no-perft" not in sys.argv else True
    sys.exit(0 if a and b else 1)
