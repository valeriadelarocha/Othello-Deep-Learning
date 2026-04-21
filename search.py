import time
import torch
import numpy as np

device = torch.device("cpu") 
shifts_tensor = torch.arange(63, -1, -1, dtype=torch.int64, device=device)
transposition_table = {}

SQUARE_WEIGHTS = [
    100, -20,  10,   5,   5,  10, -20, 100,
    -20, -50,  -2,  -2,  -2,  -2, -50, -20,
     10,  -2,  -1,  -1,  -1,  -1,  -2,  10,
      5,  -2,  -1,  -1,  -1,  -1,  -2,   5,
      5,  -2,  -1,  -1,  -1,  -1,  -2,   5,
     10,  -2,  -1,  -1,  -1,  -1,  -2,  10,
    -20, -50,  -2,  -2,  -2,  -2, -50, -20,
    100, -20,  10,   5,   5,  10, -20, 100,
]

def to_signed64(val):
    return val - 0x10000000000000000 if val & 0x8000000000000000 else val
def bitboard_to_array(bb):
    return np.unpackbits(np.array([bb], dtype='>u8').view(np.uint8)).astype(np.float32)
#
def evaluate_board(white, black, turn, env, model):
    w_moves = env.get_legal_moves(white, black).bit_count()
    b_moves = env.get_legal_moves(black, white).bit_count()
    mobility = 100 * (w_moves - b_moves) / (w_moves + b_moves + 1)
    
    w_corners = (white & 0x8100000000000081).bit_count()
    b_corners = (black & 0x8100000000000081).bit_count()
    w_x = (white & 0x0042000000004200).bit_count()
    b_x = (black & 0x0042000000004200).bit_count()
    
    pos_score = 10000 * (w_corners - b_corners) - 5000 * (w_x - b_x)
    
    w_t = bitboard_to_array(white)
    b_t = bitboard_to_array(black)
    
    inputs = np.concatenate([w_t, b_t, [np.float32(turn)]]).reshape(1, -1)
    
    nn_score = model.predict(inputs) * 1000

    total_eval = nn_score + pos_score + (5.0 * mobility)

    return total_eval if turn == 1 else -total_eval

def alpha_beta(env, white, black, turn, depth, alpha, beta, model, start_time, time_limit, is_root=False, use_nn=True):
    if time.time() - start_time > time_limit:
        raise TimeoutError("Out of time!")
    
    alpha_orig = alpha

    state_hash = hash((white, black, turn))
    p_bits = white if turn == 1 else black
    o_bits = black if turn == 1 else white
    legal_moves_bits = env.get_legal_moves(p_bits, o_bits)
    
    if legal_moves_bits == 0:
        if env.get_legal_moves(o_bits, p_bits) == 0:
            w_count = white.bit_count()
            b_count = black.bit_count()
            diff = w_count - b_count
            if diff > 0: score = 100000 + diff
            elif diff < 0: score = -100000 + diff
            else: score = 0
            return score if turn == 1 else -score, -1
        else:
            eval_score, _ = alpha_beta(env, white, black, -turn, depth-1, -beta, -alpha, model, start_time, time_limit, use_nn=use_nn)
            return -eval_score, -1

    if depth <= 0:
        if use_nn:
            return evaluate_board(white, black, turn, env, model), -1
        else:
            w_moves = env.get_legal_moves(white, black).bit_count()
            b_moves = env.get_legal_moves(black, white).bit_count()
            w_corners = (white & 0x8100000000000081).bit_count()
            b_corners = (black & 0x8100000000000081).bit_count()
            w_x = (white & 0x0042000000004200).bit_count()
            b_x = (black & 0x0042000000004200).bit_count()
            score = ((w_moves - b_moves) * 10) + ((w_corners - b_corners) * 1000) - ((w_x - b_x) * 500)
            return score if turn == 1 else -score, -1

    best_move = -1
    max_eval = -float('inf')
    moves = [i for i in range(64) if (legal_moves_bits >> i) & 1]
    
    if is_root:
        move_scores = []
        for move_idx in moves:
            move_bit = (1 << move_idx)
            new_p, new_o = env.apply_move(p_bits, o_bits, move_bit)
            new_w, new_b = (new_p, new_o) if turn == 1 else (new_o, new_p)
            
            if use_nn:
                w_t = bitboard_to_array(new_w)
                b_t = bitboard_to_array(new_b)
                
                inputs = np.concatenate([w_t, b_t, [np.float32(-turn)]]).reshape(1, -1)
                
                score = model.predict(inputs)
                move_scores.append((score if turn == 1 else -score, move_idx))   
            else:
                score = new_p.bit_count() - new_o.bit_count()
                move_scores.append((score, move_idx))
                
        move_scores.sort(reverse=True, key=lambda x: x[0])
        moves = [m[1] for m in move_scores]
    else:
        moves.sort(key=lambda idx: SQUARE_WEIGHTS[idx], reverse=True)

    if state_hash in transposition_table:
        tt_depth, tt_score, tt_flag, prev_move = transposition_table[state_hash]
        
        if tt_depth >= depth and not is_root:
            if tt_flag == 'EXACT':
                return tt_score, prev_move
            elif tt_flag == 'LOWERBOUND':
                alpha = max(alpha, tt_score)
            elif tt_flag == 'UPPERBOUND':
                beta = min(beta, tt_score)
                
            if alpha >= beta:
                return tt_score, prev_move
                
        if prev_move in moves:
            moves.remove(prev_move)
            moves.insert(0, prev_move)

    for move_idx in moves:
        move_bit = (1 << move_idx)
        new_p, new_o = env.apply_move(p_bits, o_bits, move_bit)
        new_w, new_b = (new_p, new_o) if turn == 1 else (new_o, new_p)
        
        eval_score, _ = alpha_beta(env, new_w, new_b, -turn, depth-1, -beta, -alpha, model, start_time, time_limit, use_nn=use_nn)
        eval_score = -eval_score
        if is_root:
            print(f"Move Index {move_idx} evaluated to: {eval_score:.2f}")
        if eval_score > max_eval:
            max_eval = eval_score
            best_move = move_idx
        alpha = max(alpha, eval_score)
        if beta <= alpha:
            break 
    
    if max_eval <= alpha_orig:
        flag = 'UPPERBOUND'
    elif max_eval >= beta:
        flag = 'LOWERBOUND'
    else:
        flag = 'EXACT'
        
    transposition_table[state_hash] = (depth, max_eval, flag, best_move)
    return max_eval, best_move