import socket, pickle, time, torch
from neural_net import ReversiValueNet, FastNumpyNet  # Added import
from bitboard import BitboardEnv
import search 
from search import alpha_beta

def main():
    model = ReversiValueNet().to(search.device)
    try:
        model.load_state_dict(torch.load("final_reversi_model.pt", map_location=search.device))
        print(f"Model weights loaded onto {search.device}.")
    except Exception as e:
        print(f"No weights found: {e}")
    
    model.eval()

    fast_model = FastNumpyNet(model)

    game_socket = socket.socket()
    game_socket.connect(('127.0.0.1', 33333))
    env = BitboardEnv()

    while True:
        data = game_socket.recv(4096)
        if not data: break
        turn, board = pickle.loads(data)
        if turn == 0: break

        w_bits, b_bits = env.matrix_to_bitboard(board)
        start_time = time.time()
        time_limit = 4.7 
        current_best_move = -1
        depth = 1 
        search.transposition_table.clear()

        empty_squares = 64 - (w_bits.bit_count() + b_bits.bit_count())
        use_nn_mode = empty_squares > 10
        
        if not use_nn_mode:
            print(f"--- PERFECT SOLVER ENGAGED: {empty_squares} squares left ---", flush=True)

        p_bits = w_bits if turn == 1 else b_bits
        o_bits = b_bits if turn == 1 else w_bits
        legal_bits = env.get_legal_moves(p_bits, o_bits)
        forced_moves = [i for i in range(64) if (legal_bits >> i) & 1]
        
        if len(forced_moves) == 1:
            print(f"Only 1 legal move found! Playing instantly.", flush=True)
            best_x, best_y = forced_moves[0] // 8, forced_moves[0] % 8
            game_socket.send(pickle.dumps([best_x, best_y]))
            continue  

        alpha = -float('inf')
        beta = float('inf')
        prev_score = 0

        while time.time() - start_time < time_limit:
            try:
                score, move_idx = alpha_beta(
                    env, w_bits, b_bits, turn, depth, 
                    alpha, beta, fast_model, 
                    start_time, time_limit, 
                    is_root=True, use_nn=use_nn_mode
                )
                
                if score <= alpha or score >= beta:
                    alpha = -float('inf')
                    beta = float('inf')
                    continue 
                
                if move_idx != -1: 
                    current_best_move = move_idx
                    prev_score = score 
                    
                print(f"Completed Depth {depth} in {time.time() - start_time:.2f}s", flush=True)
                
                if depth >= empty_squares:
                    break 

                elapsed_time = time.time() - start_time
                if elapsed_time > (time_limit * 0.50):
                    print("Not enough time for next depth. Returning move.", flush=True)
                    break

                alpha = prev_score - 500
                beta = prev_score + 500
                depth += 1
                
            except TimeoutError:
                print(f"--- Time Limit Triggered! Falling back to Depth {depth-1} move. ---", flush=True)
                break 
            except Exception as e:
                print(f"Search interrupted: {e}")
                break

        best_x, best_y = -1, -1
        if current_best_move != -1:
            best_x, best_y = current_best_move // 8, current_best_move % 8
        game_socket.send(pickle.dumps([best_x, best_y]))

if __name__ == '__main__':
    main()