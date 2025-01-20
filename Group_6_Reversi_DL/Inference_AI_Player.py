import numpy as np
import socket
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

# Game Constants
EMPTY = 0
BLACK_PLAYER = -1
WHITE_PLAYER = 1
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),
              ( 0, -1),          ( 0, 1),
              ( 1, -1), ( 1, 0), ( 1, 1)]

# Model Definition
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(8 * 8 + 1, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 64)

    def forward(self, x):
        x = x.to(torch.float32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Function to count trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Function to Get Valid Moves
def get_valid_moves(board, player):
    opponent = BLACK_PLAYER if player == WHITE_PLAYER else WHITE_PLAYER
    valid_moves = []

    for row in range(8):
        for col in range(8):
            if board[row][col] != EMPTY:
                continue

            for dx, dy in DIRECTIONS:
                x, y = row + dx, col + dy
                has_opponent_between = False

                while 0 <= x < 8 and 0 <= y < 8 and board[x][y] == opponent:
                    x += dx
                    y += dy
                    has_opponent_between = True

                if has_opponent_between and 0 <= x < 8 and 0 <= y < 8 and board[x][y] == player:
                    valid_moves.append((row, col))
                    break

    return valid_moves

# Main Function
def main():
    # Load the pre-trained model
    device = torch.device("cpu")
    policy_net = DQN()
    policy_net.load_state_dict(torch.load('reversi_dqn.pth', map_location=device))
    policy_net.eval()

    # Print the total number of trainable parameters
    total_params = count_parameters(policy_net)
    print(f"Total trainable parameters in the model: {total_params}")

    # Connect to the game server
    game_socket = socket.socket()
    game_socket.connect(('127.0.0.1', 33333))

    # Positional score matrix with strong corner emphasis
    positional_scores = np.array([
        [1000, -100,  10,  10,  10,  10, -100, 1000],
        [-100, -500,   5,   5,   5,   5, -500, -100],
        [  10,    5,   1,   1,   1,   1,    5,   10],
        [  10,    5,   1,   1,   1,   1,    5,   10],
        [  10,    5,   1,   1,   1,   1,    5,   10],
        [  10,    5,   1,   1,   1,   1,    5,   10],
        [-100, -500,   5,   5,   5,   5, -500, -100],
        [1000, -100,  10,  10,  10,  10, -100, 1000]
    ])

    corners = [(0, 0), (0, 7), (7, 0), (7, 7)]

    while True:
        # Receive game state
        data = game_socket.recv(4096)
        turn, board = pickle.loads(data)

        # Print the role of the player
        if turn == WHITE_PLAYER:
            print("Player role: White")
        elif turn == BLACK_PLAYER:
            print("Player role: Black")

        # Check if the game has ended
        if turn == 0:
            print("Game over.")
            game_socket.close()
            break

        # Flatten the board and append the turn indicator
        state = board.flatten()
        state = np.append(state, turn)
        state_tensor = torch.tensor(state).unsqueeze(0)

        # Get valid moves
        valid_moves = get_valid_moves(board, turn)

        # If no valid moves, pass
        if not valid_moves:
            x, y = -1, -1
        else:
            # Check for corner moves
            corner_move = next((move for move in valid_moves if move in corners), None)

            if corner_move:
                # Prioritize corner move
                x, y = corner_move
            else:
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                    mask = torch.full((64,), -float('inf'))

                    # Apply positional scores to valid moves
                    for move in valid_moves:
                        idx = move[0] * 8 + move[1]
                        mask[idx] = positional_scores[move[0], move[1]]

                    q_values += mask
                    action = q_values.argmax().item()
                    x, y = divmod(action, 8)

        # Send move to server
        game_socket.send(pickle.dumps([x, y]))

if __name__ == '__main__':
    main()
