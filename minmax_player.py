#Zijie Zhang, Sep.24/2023

import numpy as np
import socket, pickle
from reversi import reversi

def valid_moves(board, turn):
    #given the current board and turn, function calculates possible moves
    game = reversi()
    game.board = board
    moves = []
    for i in range(8):
        for j in range(8):
            if game.step(i, j, turn, False)>0:
                moves.append((i, j))
    return moves

def evaluate_board(board, turn):
    #given the current board and turn, function calculates heuristic value
    opponent = -turn
    # Mobility: Number of valid moves
    player_moves = len(valid_moves(board, turn))
    opponent_moves = len(valid_moves(board, opponent))
    
    # Corners: Control of the four corners
    corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
    player_corners = sum([1 for r, c in corners if board[r, c] == turn])
    opponent_corners = sum([1 for r, c in corners if board[r, c] == opponent])
    
    # Disk difference: The difference in number of disks
    player_disks = np.sum(board == turn)
    opponent_disks = np.sum(board == opponent)
    
    return (10 * (player_corners - opponent_corners) +
            2 * (player_moves - opponent_moves) +
            (player_disks - opponent_disks))

def minimax(board, depth, alpha, beta, maximizing_player, turn):
    #minmax algorithm
    moves = valid_moves(board, turn)
    if depth == 0 or not moves:
        return evaluate_board(board, turn), None
    
    if maximizing_player:
        max_eval = -float('inf')
        best_move = None
        for move in moves:
            new_game = reversi()
            new_game.board = board
            new_game.step(move[0], move[1], turn, True)
            eval, _ = minimax(new_game.board, depth - 1, alpha, beta, False, -turn)
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        best_move = None
        for move in moves:
            new_game = reversi()
            new_game.board = board
            new_game.step(move[0], move[1], turn, True)
            eval, _ = minimax(new_game.board, depth - 1, alpha, beta, True, -turn)
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move
    
def main():
    game_socket = socket.socket()
    game_socket.connect(('127.0.0.1', 33333))
    game = reversi()

    while True:

        #Receive play request from the server
        #turn : 1 --> you are playing as white | -1 --> you are playing as black
        #board : 8*8 numpy array
        data = game_socket.recv(4096)
        turn, board = pickle.loads(data)
        #Turn = 0 indicates game ended
        if turn == 0:
            game_socket.close()
            return
        
        #Debug info
        #print(turn)
        #print(board)

        #MinMax algorithm
        game.board = board
        #choose best move using minmax algorithm with alpha-beta pruning and depth limit of 4
        _, best_move = minimax(board, depth=4, alpha=-float('inf'), beta=float('inf'), maximizing_player=True, turn=turn)
        if not best_move:
            #Send your move to the server. Send (x,y) = (-1,-1) to tell the server you have no hand to play
            game_socket.send(pickle.dumps([-1,-1]))    
        else:
            game_socket.send(pickle.dumps([best_move[0],best_move[1]]))
        
if __name__ == '__main__':
    main()