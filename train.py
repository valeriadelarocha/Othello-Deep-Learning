import os
import time
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import search
from search import alpha_beta
import agent 
from agent import ReversiValueNet, BitboardEnv, alpha_beta, FastNumpyNet


gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu_device = torch.device("cpu")

search.device = cpu_device 
search.shifts_tensor = search.shifts_tensor.to(cpu_device)

BATCH_SIZE = 4096 
REPLAY_BUFFER_SIZE = 150000 
LEARNING_RATE = 0.0001
EPISODES = 800  
TRAIN_EVERY = 5    
SEARCH_DEPTH = 5   

shifts_tensor = torch.arange(63, -1, -1, dtype=torch.int64, device=gpu_device)
memory = deque(maxlen=REPLAY_BUFFER_SIZE)

def get_symmetries(white, black):
    """Returns 8 symmetric versions of the board."""
    syms = set()
    w_arr = np.unpackbits(np.array([white], dtype='>u8').view(np.uint8)).reshape(8, 8)
    b_arr = np.unpackbits(np.array([black], dtype='>u8').view(np.uint8)).reshape(8, 8)

    for i in range(4):
        w_rot = np.rot90(w_arr, i)
        b_rot = np.rot90(b_arr, i)
        syms.add((int.from_bytes(np.packbits(w_rot).tobytes(), 'big'),
                  int.from_bytes(np.packbits(b_rot).tobytes(), 'big')))
        
        w_flip = np.fliplr(w_rot)
        b_flip = np.fliplr(b_rot)
        syms.add((int.from_bytes(np.packbits(w_flip).tobytes(), 'big'),
                  int.from_bytes(np.packbits(b_flip).tobytes(), 'big')))
                  
    return list(syms)

def save_game_to_memory(game_history, winner):
    for w, b, t in game_history:
        for sym_w, sym_b in get_symmetries(w, b):
            memory.append((sym_w, sym_b, t, winner))

def train_step(model, optimizer, criterion):
    if len(memory) < BATCH_SIZE:
        return None
    
    batch = random.sample(memory, BATCH_SIZE)
    w_batch, b_batch, t_batch, y_batch = zip(*batch)
    
    # Send directly to GPU
    w_tensor = torch.from_numpy(np.array(w_batch, dtype=np.uint64).astype(np.int64)).unsqueeze(1).to(gpu_device)
    b_tensor = torch.from_numpy(np.array(b_batch, dtype=np.uint64).astype(np.int64)).unsqueeze(1).to(gpu_device)
    t_tensor = torch.tensor(t_batch, dtype=torch.float32).unsqueeze(1).to(gpu_device)
    Y = torch.tensor(y_batch, dtype=torch.float32).unsqueeze(1).to(gpu_device)
    
    w_bits = ((w_tensor >> shifts_tensor) & 1).float()
    b_bits = ((b_tensor >> shifts_tensor) & 1).float()
    
    X = torch.cat([w_bits, b_bits, t_tensor], dim=1)
    
    optimizer.zero_grad()
    outputs = model.layers(X) 
    loss = criterion(outputs, Y)
    loss.backward()
    optimizer.step()
    
    return loss.item()

def start_training():
    model = ReversiValueNet().to(cpu_device)
    checkpoint_path = "reversi_model_checkpoint.pt"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded existing model.")
        
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()
    env = BitboardEnv()
    
    fast_model = FastNumpyNet(model) 
    print(f"\nStarting Minimax Self-Play...")
    print(f"Simulations running on: {cpu_device}")
    print(f"Batch Training running on: {gpu_device}")

    for ep in range(EPISODES):
        epsilon = max(0.01, 0.10 - (ep / EPISODES) * 0.09)
        game_history = []
        w_bits, b_bits = 0x0000000810000000, 0x0000001008000000 
        turn = 1
        pass_count = 0
        
        while pass_count < 2: 
            p_bits = w_bits if turn == 1 else b_bits
            o_bits = b_bits if turn == 1 else w_bits
            
            legal_bits = env.get_legal_moves(p_bits, o_bits)
            moves = [i for i in range(64) if (legal_bits >> i) & 1]
            
            if not moves:
                pass_count += 1
                turn *= -1
                continue
                
            pass_count = 0
            game_history.append((w_bits, b_bits, turn))

            if random.random() < epsilon: 
                move_idx = random.choice(moves)
            else:
                start_time = time.time()
                
                search.transposition_table.clear() 
                
                _, move_idx = alpha_beta(
                    env, w_bits, b_bits, turn, SEARCH_DEPTH, 
                    -float('inf'), float('inf'), fast_model, 
                    start_time, time_limit=999.0, is_root=True, use_nn=True
                )
            
            if move_idx == -1:
                move_idx = random.choice(moves)

            move_bit = (1 << move_idx)
            if turn == 1:
                w_bits, b_bits = env.apply_move(w_bits, b_bits, move_bit)
            else:
                b_bits, w_bits = env.apply_move(b_bits, w_bits, move_bit)
            
            turn *= -1

        w_count = bin(w_bits).count('1')
        b_count = bin(b_bits).count('1')
        winner = 1 if w_count > b_count else -1 if b_count > w_count else 0
        
        save_game_to_memory(game_history, winner)
        
        print(f"Game {ep+1} Finished | Result: W{w_count}-B{b_count} | Buffer: {len(memory)}/{REPLAY_BUFFER_SIZE}", flush=True)
        
        if ep % TRAIN_EVERY == 0 and len(memory) >= BATCH_SIZE:
            model.to(gpu_device) 
            loss = train_step(model, optimizer, criterion)
            model.to(cpu_device) 
            fast_model = FastNumpyNet(model)
            if loss is not None:
                print(f"   >>> Training Step | Loss: {loss:.4f}")

        if (ep + 1) % 100 == 0:
            torch.save(model.state_dict(), checkpoint_path)

    torch.save(model.state_dict(), "final_reversi_model.pt")
    print("Training Complete. Final model saved.")

if __name__ == "__main__":
    start_training()